"""订单系统的 prod 实现 —— 待接入。

v1 只跑通 eval 数据源，prod 实现留桩。接入时在这里发 HTTP：

    POST {ORDER_SVC}/refunds            Idempotency-Key: {request_id}
    POST {ORDER_SVC}/refund-denials     Idempotency-Key: {request_id}

资格判定走 `services/rule/`，不在这一侧。这里只做协议映射。
"""

from services.order.protocol import RefundReceipt

_UNAVAILABLE = "prod 订单系统未接入：v1 仅支持 request_source=eval"


class ProdOrderService:
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
