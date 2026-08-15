"""政策检索服务的接口与数据模型。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PolicySection:
    section: str
    text: str
    score: float = 0.0
    """相似度分。eval 实现返回写死条款，无检索过程，恒为 0。"""


class RagService(Protocol):
    def search_policy(self, query: str, top_k: int = 4) -> list[PolicySection]:
        """检索适用的退款政策条款。"""
        ...
