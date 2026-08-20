"""订单系统的 prod SQLite 实现。"""

from services import prod_store
from services.order.protocol import RefundReceipt


class ProdOrderService:
    def execute_refund(
        self,
        order_id: str,
        acting_user: str,
        amount: float,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        with prod_store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._replay(db, idempotency_key)
            if existing:
                return existing

            order = db.execute(
                "SELECT customer_id, refunded FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if order is None or order["customer_id"] != acting_user:
                raise ValueError(f"订单 {order_id} 不存在")
            if order["refunded"]:
                raise ValueError(f"订单 {order_id} 已退款")

            receipt = RefundReceipt(self._next_receipt(db, "R"), amount)
            db.execute("UPDATE orders SET refunded = 1 WHERE order_id = ?", (order_id,))
            self._insert_decision(
                db, "批准", receipt, order_id, reason, acting_user, idempotency_key
            )
            return receipt

    def record_denial(
        self,
        order_id: str,
        acting_user: str,
        reason: str,
        idempotency_key: str,
    ) -> RefundReceipt:
        with prod_store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._replay(db, idempotency_key)
            if existing:
                return existing

            receipt = RefundReceipt(self._next_receipt(db, "D"))
            self._insert_decision(
                db, "拒绝", receipt, order_id, reason, acting_user, idempotency_key
            )
            return receipt

    @staticmethod
    def _replay(db, idempotency_key: str) -> RefundReceipt | None:
        if not idempotency_key:
            return None
        row = db.execute(
            "SELECT receipt_no, amount FROM refund_decisions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return RefundReceipt(row["receipt_no"], row["amount"]) if row else None

    @staticmethod
    def _next_receipt(db, prefix: str) -> str:
        row = db.execute(
            "SELECT COALESCE(MAX(id), 0) + 9000 AS number FROM refund_decisions"
        ).fetchone()
        return f"{prefix}{row['number']}"

    @staticmethod
    def _insert_decision(
        db,
        decision: str,
        receipt: RefundReceipt,
        order_id: str,
        reason: str,
        actor: str,
        idempotency_key: str,
    ) -> None:
        db.execute(
            """
            INSERT INTO refund_decisions
                (decision, receipt_no, order_id, amount, reason, actor, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision,
                receipt.receipt_no,
                order_id,
                receipt.amount,
                reason,
                actor,
                idempotency_key or None,
            ),
        )
