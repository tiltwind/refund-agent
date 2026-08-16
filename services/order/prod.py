"""订单系统的 prod 实现 —— 待接入。

v1 只跑通 eval 数据源，prod 实现留桩。接入时在这里发 HTTP：

    POST {ORDER_SVC}/orders/{order_id}/refund-eligibility   ← 规则引擎在对侧
    POST {ORDER_SVC}/refunds            Idempotency-Key: {request_id}
    POST {ORDER_SVC}/refund-denials     Idempotency-Key: {request_id}

**规则引擎不要搬到这一侧来。** 窗口计算要用签收时间（数据在订单库）、
授权判定必须在数据所有者一侧、规则变更由订单团队独立发版 —— 这三条决定了
它必须留在订单系统里（1-architecture 第一章）。这里只做协议映射。
"""

from services.order.protocol import EligibilityResult, RefundReceipt

_UNAVAILABLE = "prod 订单系统未接入：v1 仅支持 request_source=eval"


class ProdOrderService:
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        raise NotImplementedError(_UNAVAILABLE)

    def execute_refund(
        self,
        order_id: str,
        acting_user: str,
        amount: float,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        raise NotImplementedError(_UNAVAILABLE)

    def record_denial(
        self,
        order_id: str,
        acting_user: str,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        raise NotImplementedError(_UNAVAILABLE)
