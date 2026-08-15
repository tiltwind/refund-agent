"""客户档案服务的 eval 实现 —— 读 evals/data/customers.json。"""

from services import eval_store
from services.customer.protocol import CustomerProfile, OrderBrief
from services.errors import EvalDataMissError


class EvalCustomerService:
    def get_profile(self, customer_id: str) -> CustomerProfile:
        row = eval_store.customers().get(customer_id)
        if row is None:
            # 静默返回空档案会让用例带着「查无此人」这个看似合理的错误结论通过。
            # customer_id 来自 context（评估流水线构造），查不到就是数据覆盖不足。
            raise EvalDataMissError(
                f"eval 数据缺少客户 {customer_id}，请补充 evals/data/customers.json"
            )

        orders = [
            OrderBrief(
                order_id=oid,
                product=o["product"],
                category=o["category"],
                price=o["price"],
                signed_days_ago=o["signed_days_ago"],
                refunded=o["refunded"],
            )
            for oid, o in eval_store.orders().items()
            if o["customer_id"] == customer_id
        ]
        return CustomerProfile(
            customer_id=customer_id,
            name=row["name"],
            level=row["level"],
            register_date=row["register_date"],
            refund_count_90d=row["refund_count_90d"],
            orders=orders,
        )
