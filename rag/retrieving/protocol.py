"""政策检索服务的接口与数据模型。"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PolicySection:
    """一条装配好的证据。

    字段不只是「文本 + 分数」—— 检索结果必须同时带回**内容、来源、时间、
    相关性理由**，缺一项都会在下游付出代价：
    - 没有来源，答复里引用不了原文、事后审计追不到出处；
    - 没有生效日期，模型无从判断这条还算不算数；
    - 没有相关性理由和分数，线上出坏 case 时无法区分「是召回错了」还是
      「是模型答错了」—— 这是调试检索质量唯一的抓手。
    """

    section: str
    """人读的来源名：`P02 星辰优选售后服务与退换货政策 > 第二条 退货窗口`。"""
    text: str
    """条款原文（父块回填后的完整小节）。"""
    score: float = 0.0
    """重排后的最终分。"""
    doc_id: str = ""
    layer: str = ""
    """law | platform —— 答复消费者引用 platform，判断条款效力才引 law。"""
    doc_type: str = ""
    effective_date: str = ""
    source_path: str = ""
    """仓库内相对路径，做引用定位。"""
    reason: str = ""
    """相关性理由：这条是被哪一路召回、被哪个子查询命中的。"""


@dataclass
class RetrievalTrace:
    """一次检索的中间产物 —— 每一步的输入输出都记下来。

    没有这些日志，坏 case 只能靠猜：明明有条款却没召回，到底是被生效日期
    过滤掉了、是两路都没排进 TopK、还是重排把它压下去了？这三种情况的修法
    完全不同，而它们在最终结果里长得一模一样。

    `steps` 是人读的，两个 ID 序列是机器读的。Recall@k 的真值是 chunk_id 集合
    比对，从 `steps` 里那句「top3=[...]」正则抠 ID 也能算，但那句话的措辞属于
    日志，改一个字打分器就静默判负 —— 指标依赖的东西要单独存。
    """

    query: str = ""
    steps: list[tuple[str, str]] = field(default_factory=list)

    candidate_ids: list[str] = field(default_factory=list)
    """召回融合后的候选，按 RRF 序。`recall@10` 读它。"""
    evidence_ids: list[str] = field(default_factory=list)
    """重排后过阈值的证据，按最终分降序。`recall@3` / `recall@1` 读它。"""

    def record(self, step: str, detail: str) -> None:
        self.steps.append((step, detail))

    def render(self) -> str:
        lines = [f"检索链路 query={self.query!r}"]
        lines += [f"  {i}. {name}：{detail}" for i, (name, detail) in enumerate(self.steps, 1)]
        return "\n".join(lines)


class RagService(Protocol):
    def search_policy(self, query: str, top_k: int = 4) -> list[PolicySection]:
        """检索适用的退款政策条款。"""
        ...
