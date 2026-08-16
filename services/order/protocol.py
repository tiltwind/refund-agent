"""订单系统的接口与数据模型：退款执行。

资格判定不在这里，在 `services/rule/` —— 规则口径的变更频率远高于订单数据与
资金链路，拆开两边才能各自独立发版（1-architecture 第一章）。
订单系统只保留终局动作：打款与拒绝落库。
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RefundReceipt:
    """终局动作的回执 —— 单号只有真正调用了执行接口才拿得到（1-architecture 第二章）。"""

    receipt_no: str
    amount: float = 0.0


class OrderService(Protocol):
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
