"""政策文档切片 —— doc/policy/**/*.md → 可入库的父子块。

对外只暴露一个入口：`chunk_document(path, tokens, encode)`。
切分策略与参数的完整理由见 policy.py 的模块 docstring。
"""

from knowledge.chunking.model import Chunk, DocMeta
from knowledge.chunking.policy import CHILD_MAX_TOKENS, CHILD_TARGET_TOKENS, chunk_document

__all__ = [
    "CHILD_MAX_TOKENS",
    "CHILD_TARGET_TOKENS",
    "Chunk",
    "DocMeta",
    "chunk_document",
]
