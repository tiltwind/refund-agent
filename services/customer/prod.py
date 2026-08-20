"""客户档案服务的 prod SQLite 实现。"""

from services import prod_store
from services.customer.protocol import CustomerProfile, OrderBrief


class ProdCustomerService:
    def get_profile(self, customer_id: str) -> CustomerProfile:
        with prod_store.connect() as db:
            customer = db.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
            if customer is None:
                raise ValueError(f"客户 {customer_id} 不存在")
            rows = db.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_id",
                (customer_id,),
            ).fetchall()

        orders = [
            OrderBrief(
                order_id=row["order_id"],
                product=row["product"],
                category=row["category"],
                price=row["price"],
                signed_days_ago=row["signed_days_ago"],
                refunded=bool(row["refunded"]),
            )
            for row in rows
        ]
        return CustomerProfile(
            customer_id=customer["customer_id"],
            name=customer["name"],
            level=customer["level"],
            register_date=customer["register_date"],
            refund_count_90d=customer["refund_count_90d"],
            orders=orders,
        )
