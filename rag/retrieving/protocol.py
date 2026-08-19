"""政策检索服务的接口与数据模型。"""

from dataclasses import dataclass, field
from typing import Protocol


class RetrievalError(RuntimeError):
    """检索链路无法提供可用证据。

    带上这次检索的 `trace`：链路交不出证据时，最先要回答的是「召回层到底捞到
    没有」—— 捞到了被阈值砍光和压根没捞到，修法完全不同。异常里不带中间产物，
    调用方（评测、线上错误处理）就只剩一句错误消息可看。
    """

    def __init__(self, message: str, trace: "RetrievalTrace | None" = None) -> None:
        super().__init__(message)
        self.trace = trace


class NoCandidatesError(RetrievalError):
    """召回层没有任何候选，通常表示知识库或过滤配置故障。"""


class NoEvidenceError(RetrievalError):
    """候选存在，但重排/装配后没有可交付证据。"""


@dataclass
class PolicySection:
    """一条装配好的证据。

    字段不只是「文本 + 分数」—— 检索结果必须同时带回**内容、来源、时间、
    相关性理由**，每一项都对着下游一件具体的事：
    - 来源：答复里引用原文，事后审计追出处；
    - 生效日期：模型据此判断这条还算不算数；
    - 相关性理由和分数：线上出坏 case 时区分「是召回错了」还是「是模型答错了」
      —— 这是调试检索质量唯一的抓手。
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

    `steps` 是人读的，两个 ID 序列是机器读的：装配那一步要拿证据 ID，上报的
    span 也要它们 —— 从 `steps` 里那句「top3=[...]」正则抠 ID 也能拿到，但那句话
    的措辞属于日志，改一个字下游就静默出错。
    """

    query: str = ""
    steps: list[tuple[str, str]] = field(default_factory=list)

    candidate_ids: list[str] = field(default_factory=list)
    """召回融合后的候选，按 RRF 序。"""
    evidence_ids: list[str] = field(default_factory=list)
    """重排后过阈值的证据，按最终分降序。"""

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
