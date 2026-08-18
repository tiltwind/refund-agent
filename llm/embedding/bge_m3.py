"""BGE-M3 稠密嵌入 —— 本地跑，灌库（rag/index/）与检索（rag/retrieving/）共用。

**为什么只取 dense**：BGE-M3 能同时产出 dense / sparse / colbert 三种表示，
但本项目的字面匹配一路交给 Milvus 2.5 的原生 BM25（见 rag/index/seed_milvus.py）。
BM25 有成熟的分词器、可解释的打分、不需要额外加载 sparse_linear 权重，
在「条款号、金额、`7 天`、`max_idle_conns` 这类精确 term」上正是它的主场。
两路各司其职，再在应用层显式做 RRF 融合（rag/retrieving/pipeline/recall.py）。

**为什么不用 FlagEmbedding / sentence-transformers**：直接用 transformers
是为了把 pooling 与截断长度摆在明面上。这两件事一旦在灌库与检索之间不一致，
检索结果不会报错，只会悄悄变差 —— 这是 RAG 里最难定位的一类故障。

**为什么没有降级实现**：换嵌入模型等于换向量空间，TopK 排序整体重来。
拿不到模型时必须显式失败，不能悄悄退回哈希嵌入那种「只认字面」的兜底 ——
那会让检索看起来在工作，实际上召回的条款与问题无关。
"""

import os
from functools import lru_cache

MODEL_ID = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")

DIMENSION = 1024
"""BGE-M3 dense 向量维度。它与 max_length 无关 —— 输入多长，输出恒为 1024 维。"""

MAX_LENGTH = int(os.getenv("BGE_M3_MAX_LENGTH", "1024"))
"""输入截断长度。

BGE-M3 标称上限 8192，这里只开到 1024，两个理由：
1. 子块目标 320 token、硬上限 512（rag/chunking/policy.py），1024 是三倍
   余量，`truncated == 0` 有保证；
2. 长度外推衰减是真实存在的 —— 标称 8192 不代表在 8192 上的表示质量与 512
   一致，长文本 mean/CLS pooling 后语义会被摊平。开到用不上的长度没有收益，
   只增加显存与耗时。

灌库与检索必须用同一个值：截断长度不一致会让 query 与 passage 的编码不对称。
"""

BATCH_SIZE = int(os.getenv("BGE_M3_BATCH_SIZE", "8"))


class BgeM3Embedder:
    """加载一次、进程内复用。构造即加载模型，失败就抛。"""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "BGE-M3 需要 torch 与 transformers：pip install -r requirements.txt"
            ) from exc

        from llm.device import pick_device

        self._torch = torch
        self.model_id = model_id
        self.device = device or pick_device()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"加载嵌入模型「{model_id}」失败（{type(exc).__name__}: {exc}）。\n"
                "首次运行需要联网下载约 2.2 GB 权重；已离线缓存过则用环境变量指向本地目录：\n"
                "  export BGE_M3_MODEL=/path/to/bge-m3\n"
                "国内网络可先设 export HF_ENDPOINT=https://hf-mirror.com"
            ) from exc

        self.model.eval().to(self.device)

        # 模型自己报维度，不写死 —— 换模型后维度对不上会在建表时直接炸，
        # 而不是灌进一个维度错配的 collection 再在检索时返回一堆不相关条款。
        hidden = int(self.model.config.hidden_size)
        if hidden != DIMENSION:
            raise RuntimeError(
                f"模型 {model_id} 的向量维度为 {hidden}，与本项目约定的 {DIMENSION} 不符；"
                "换嵌入模型必须同时重建 collection 并重跑检索基线"
            )

    # ── 长度 ──────────────────────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """按**本模型的 tokenizer** 计 token 数，切分时用。

        切分器天然按字符计数，而截断限制按 token 计 —— 中文 1 字 ≈ 1~1.5 token
        （XLM-R BPE），直接拿字符数当 token 数会静默超限。所以切分的长度函数
        必须是这个方法，不能是 len()。
        """
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def truncation_report(self, texts: list[str]) -> dict:
        """入库前把「静默截断」变成显式数字。验收标准：truncated == 0。"""
        lengths = sorted(self.count_tokens(t) for t in texts)
        if not lengths:
            return {"n": 0, "p50": 0, "p99": 0, "max": 0, "limit": MAX_LENGTH, "truncated": 0}
        return {
            "n": len(lengths),
            "p50": lengths[len(lengths) // 2],
            "p99": lengths[min(int(len(lengths) * 0.99), len(lengths) - 1)],
            "max": lengths[-1],
            "limit": MAX_LENGTH,
            "truncated": sum(1 for n in lengths if n > MAX_LENGTH),
        }

    # ── 编码 ──────────────────────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> list[list[float]]:
        from llm.device import inference_lock

        torch = self._torch
        out: list[list[float]] = []
        # 前向必须串行：MPS 后端不是线程安全的，并发跑会段错误（llm/device.py）
        with inference_lock(), torch.inference_mode():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                # BGE-M3 的 dense 表示取 [CLS]（第 0 个 token），不是 mean pooling。
                # 取错 pooling 不会报错，只会让相似度整体失真。
                vecs = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
                out.extend(vecs.float().cpu().tolist())
        return out

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_query(self, text: str) -> list[float]:
        # 与 passage 走**同一条编码路径**：同模型、同 pooling、同归一化、同截断。
        # BGE-M3 不需要 instruction prefix（bge-v1.5 系列才需要），两边都不加。
        return self._encode([text])[0]


@lru_cache(maxsize=1)
def embedder() -> BgeM3Embedder:
    """进程内单例。首次调用时才加载，让不检索的路径不必等模型。"""
    return BgeM3Embedder()
