"""嵌入模型 —— 灌库（knowledge/seed_milvus.py）与检索（milvus.py）共用同一份。

两边必须用**同一个模型**：向量空间不同则检索结果毫无意义。所以维度由模型
自己报（`dimension()`），collection 建表时按它来，换模型后维度对不上会在
灌库/检索时直接报错，而不是悄悄返回一堆不相关的条款。

有 OPENAI_API_KEY 就用 OpenAI embeddings，否则回退到哈希嵌入（仅字面匹配、
无语义理解，但足够本地跑通链路）。生产环境应固定成一个明确的模型版本 ——
embedding 换版本会让向量空间整体偏移，TopK 排序全变。
"""

import hashlib
import math
import os

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """离线兜底嵌入：字符 bigram 哈希 + L2 归一化。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        grams = [text[i : i + 2] for i in range(len(text) - 1)] or [text]
        for g in grams:
            idx = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings() -> Embeddings:
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            )
            # 启动探针：尽早暴露「网关不支持 /embeddings」这类问题，
            # 而不是等到第一次检索时才在业务链路里炸
            embeddings.embed_query("ping")
            return embeddings
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] OpenAI embeddings 不可用（{type(exc).__name__}），回退到哈希嵌入")
    return HashEmbeddings()


def dimension(embeddings: Embeddings) -> int:
    """探一次向量维度 —— 建 collection 要用。"""
    return len(embeddings.embed_query("dimension probe"))
