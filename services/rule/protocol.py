"""规则服务的接口与数据模型：退款资格判定。

**规则引擎是一个独立的下游服务，不在 Agent 服务里**（1-architecture 第一章）：
退款规则的变更频率远高于订单数据与资金链路，独立成服务才能独立发版；
判定与执行分开，也让「谁能改判定口径」和「谁能动钱」是两拨权限。

归属校验仍在数据所有者一侧：prod 规则服务带 `acting_user` 去订单系统取数，
取不到就是「订单不存在」，Agent 无从绕过（2-design 1.6）。
"""

from dataclasses import dataclass
from typing import Literal, Protocol

Verdict = Literal["通过", "不通过", "需补充"]


@dataclass
class EligibilityResult:
    """规则引擎的判定结论 —— 这是退款决策的**唯一依据**，模型不得推翻。"""

    verdict: Verdict
    reason: str
    """判定说明，直接给模型读，也是答复里引用的依据。"""
    refundable_amount: float = 0.0
    """可退金额，仅 verdict == 通过 时有意义。"""


class RuleService(Protocol):
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        """判定退款资格。acting_user 由 Context 注入，用于归属校验。"""
        ...
