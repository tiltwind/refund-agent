"""订单系统的 eval 实现 —— 读 evals/data/orders.json，只做终局动作。

**终局动作必须 stub**：execute_refund 只记录调用意图，不发起任何打款，
断言「是否调用 + 参数是否正确」即可（2-design 6.3）。

资格判定在 `services/rule/eval.py`。
"""

from services import eval_store
from services.order.protocol import RefundReceipt


class EvalOrderService:
    def execute_refund(
        self,
        order_id: str,
        acting_user: str,
        amount: float,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        existing = self._replay(idempotency_key)
        if existing:
            return existing

        order = eval_store.orders().get(order_id)
        if not order or order["customer_id"] != acting_user:
            raise ValueError(f"订单 {order_id} 不存在")
        if order["refunded"]:
            raise ValueError(f"订单 {order_id} 已退款")

        # stub：只置状态、只记流水，不调用任何打款接口
        order["refunded"] = True
        log = eval_store.decision_log()
        receipt = RefundReceipt(f"R{9000 + len(log)}", amount)
        log.append(
            {
                "decision": "批准",
                "receipt_no": receipt.receipt_no,
                "order_id": order_id,
                "amount": amount,
                "reason": reason,
                "actor": acting_user,
                "idempotency_key": idempotency_key,
            }
        )
        return receipt

    def record_denial(
        self,
        order_id: str,
        acting_user: str,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        existing = self._replay(idempotency_key)
        if existing:
            return existing

        log = eval_store.decision_log()
        receipt = RefundReceipt(f"D{9000 + len(log)}")
        log.append(
            {
                "decision": "拒绝",
                "receipt_no": receipt.receipt_no,
                "order_id": order_id,
                "amount": 0.0,
                "reason": reason,
                "actor": acting_user,
                "idempotency_key": idempotency_key,
            }
        )
        return receipt

    @staticmethod
    def _replay(idempotency_key: str) -> RefundReceipt | None:
        """同一幂等键返回同一个单号，而不是重复落一笔（2-design 第四章）。"""
        if not idempotency_key:
            return None
        for row in eval_store.decision_log():
            if row["idempotency_key"] == idempotency_key:
                return RefundReceipt(row["receipt_no"], row["amount"])
        return None
