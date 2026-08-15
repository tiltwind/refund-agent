"""嵌入模型 —— 当前只有 BGE-M3 一个实现。"""

from llm.embedding.bge_m3 import BgeM3Embedder, embedder

__all__ = ["BgeM3Embedder", "embedder"]
