"""bge-reranker-v2-m3 交叉编码重排 —— 本地跑。

与嵌入的**双塔**编码不同：双塔把 query 和 passage 各自压成一个向量再算距离，
判断的是「主题像不像」；cross-encoder 让两边的 token 互相 attend，判断的是
「这段文字回没回答这个问题」。放到本项目的语料上，差别很具体 ——
P02 第二条（退货窗口）和 P07 第三条（会员权益延长）主题几乎一样，
但只有前者能回答「签收 10 天还能退吗」。这是召回阶段区分不出来的。

**这个模型是可选的**（`REFUND_AGENT_RERANK=off` 关闭，或权重下载失败自动降级）。
关掉不会让链路失败，只是重排退化为「融合分 + 效力位阶加权」——
少一段精排，多一点噪声进上下文。降级发生时会打一行 warn，别让它悄悄发生。
"""

import os
from functools import lru_cache

MODEL_ID = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = int(os.getenv("BGE_RERANKER_MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8"))

ENABLED = os.getenv("REFUND_AGENT_RERANK", "on").lower() not in ("off", "0", "false")


class BgeReranker:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from llm.device import pick_device

        self._torch = torch
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval().to(self.device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """返回每个 passage 的相关性分，已用 sigmoid 归一化到 (0, 1)。

        归一化不是为了好看：重排分要和效力位阶等特征做加权求和
        （rag/retrieving/pipeline/rerank.py），原始 logit 无上下界，
        直接加权会让权重完全失去意义。
        """
        if not passages:
            return []
        from llm.device import inference_lock

        torch = self._torch
        out: list[float] = []
        # 与嵌入共用一把锁：同一块 GPU，串行不慢，并发会段错误（llm/device.py）
        with inference_lock(), torch.inference_mode():
            for i in range(0, len(passages), BATCH_SIZE):
                batch = passages[i : i + BATCH_SIZE]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.view(-1).float()
                out.extend(torch.sigmoid(logits).cpu().tolist())
        return out


@lru_cache(maxsize=1)
def reranker() -> BgeReranker | None:
    """拿不到就返回 None —— 调用方据此降级，不抛异常打断检索。

    这里的取舍与嵌入模型相反：嵌入拿不到就必须失败（向量空间对不上，
    检索结果无意义），重排拿不到只是排序变粗，融合分仍然可用。
    """
    if not ENABLED:
        return None
    try:
        return BgeReranker()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] 重排模型「{MODEL_ID}」不可用（{type(exc).__name__}: {exc}），"
            "本次检索降级为「融合分 + 效力位阶加权」，精排缺失"
        )
        return None
