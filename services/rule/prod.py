"""规则服务的 prod SQLite 实现。"""

from services import prod_store
from services.rule.protocol import EligibilityResult

MAX_REFUND_WINDOW = 15
NON_REFUNDABLE_CATEGORIES = {"生鲜", "定制", "虚拟商品"}
HIGH_RISK_REFUND_COUNT = 3


class ProdRuleService:
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        with prod_store.connect() as db:
            order = db.execute(
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (order_id, acting_user),
            ).fetchone()
            if order is None:
                return EligibilityResult("不通过", f"订单 {order_id} 不存在")
            if order["refunded"]:
                return EligibilityResult("不通过", f"订单 {order_id} 已退款，不可重复申请")
            customer = db.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (acting_user,)
            ).fetchone()

        refund_count = customer["refund_count_90d"]
        if refund_count > HIGH_RISK_REFUND_COUNT:
            return EligibilityResult(
                "不通过",
                f"客户近 90 天退款 {refund_count} 次，超过 {HIGH_RISK_REFUND_COUNT} 次阈值，"
                "属高风险账户，自动退款通道已关闭，需转人工客服审核（1-3 个工作日）",
            )
        if order["category"] in NON_REFUNDABLE_CATEGORIES:
            return EligibilityResult(
                "不通过", f"商品类目「{order['category']}」不支持任何形式的退款"
            )

        days = order["signed_days_ago"]
        if days > MAX_REFUND_WINDOW:
            return EligibilityResult(
                "不通过",
                f"订单已签收 {days} 天，已超出所有类型的退货窗口"
                f"（最宽为「质量问题」{MAX_REFUND_WINDOW} 天）",
            )
        if not reason_type:
            return EligibilityResult(
                "需补充",
                "该订单未命中任何硬否决规则，接下来的判定取决于退款原因。"
                "请向用户确认是「无理由」还是「质量问题」，再次调用本工具。",
            )
        if reason_type != "质量问题" and not item_condition:
            return EligibilityResult(
                "需补充",
                "无理由退货的判定取决于商品状态。请向用户确认商品是"
                "「未拆封」「已拆封」还是「已使用」，再次调用本工具。",
            )

        window = 15 if reason_type == "质量问题" else (
            15 if customer["level"] == "金牌会员" else 7
        )
        if days > window:
            return EligibilityResult(
                "不通过",
                f"订单已签收 {days} 天，超出「{reason_type}」退货窗口"
                f"（{customer['level']} {window} 天）",
            )
        if reason_type != "质量问题" and item_condition != "未拆封":
            return EligibilityResult(
                "不通过",
                f"商品状态为「{item_condition}」，无理由退货要求未拆封、不影响二次销售",
            )
        return EligibilityResult(
            "通过",
            f"订单 {order_id} 符合「{reason_type}」退款条件"
            f"（签收 {days} 天 ≤ 窗口 {window} 天，商品状态：{item_condition or '不限'}）",
            refundable_amount=order["price"],
        )
