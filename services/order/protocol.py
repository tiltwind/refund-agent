"""订单系统的接口与数据模型：资格判定 + 退款执行。

**规则引擎在订单系统一侧，不在 Agent 服务里**（README 第二章）：
数据在那边（窗口计算要用签收时间）、授权判定必须在数据所有者一侧、
规则变更由订单团队独立发版而不必动 Agent。
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


@dataclass
class RefundReceipt:
    """终局动作的回执 —— 单号只有真正调用了执行接口才拿得到（README 第三章）。"""

    receipt_no: str
    amount: float = 0.0


class OrderService(Protocol):
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        """判定退款资格。acting_user 由 Context 注入，用于归属校验。"""
        ...

    def execute_refund(
        self,
        order_id: str,
        acting_user: str,
        amount: float,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        """执行退款打款。idempotency_key 用 request_id，同键不重复打款。"""
        ...

    def record_denial(
        self,
        order_id: str,
        acting_user: str,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        """记录拒绝决策并落库，返回可供用户查询/申诉的受理编号。"""
        ...
