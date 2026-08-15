"""订单系统的 eval 实现 —— 读 evals/data/orders.json，内含**规则引擎副本**。

线上这套规则跑在订单系统里（README 第二章）；离线评估连不上它，于是在这里
维护一份等价副本。两条纪律：

1. **规则变更时两边必须同步**，否则离线回归会用旧口径给出「与事实相反」的
   报告 —— 这正是 README 9.1 里 ⑤ 那条反向箭头要防的事：用例挂了，先确认
   是 Agent 错了还是评估侧的口径滞后了。
2. **终局动作必须 stub**：execute_refund 只记录调用意图，不发起任何打款，
   断言「是否调用 + 参数是否正确」即可（README 9.3）。
"""

from services import eval_store
from services.order.protocol import EligibilityResult, RefundReceipt

# 最宽的退货窗口：「质量问题」15 天，且不分会员等级。连它都超了，就说明
# 无论用户回答什么理由、什么商品状态，判定都不会通过 —— 不必再追问。
MAX_REFUND_WINDOW = 15

# 不支持退款的类目（与「特殊类目」条款一致）
NON_REFUNDABLE_CATEGORIES = {"生鲜", "定制", "虚拟商品"}

# 高风险阈值（与「高风险用户」条款一致），判定条件是 > 阈值
HIGH_RISK_REFUND_COUNT = 3


class EvalOrderService:
    # ── 资格判定 ──────────────────────────────────────────────────────────
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        order = eval_store.orders().get(order_id)

        # 归属校验（README 4.6）：不存在 与 不属于你 返回同一句话 ——
        # 区分开会泄露「这个订单号是否存在」。这里订单号来自对话文本，模型
        # 完全可能编一个，所以「查不到」是合法业务结论，不是 eval 数据缺失。
        if not order or order["customer_id"] != acting_user:
            return EligibilityResult("不通过", f"订单 {order_id} 不存在")

        if order["refunded"]:
            return EligibilityResult("不通过", f"订单 {order_id} 已退款，不可重复申请")

        customer = eval_store.customers()[order["customer_id"]]

        # ── 第一段：与 reason_type / item_condition 无关的硬否决，命中即定案 ──
        # 这一段把「何时该追问用户」从模型的自由裁量收回到规则里：这些情形下
        # 无论用户怎么回答都不会通过，追问只会白白拖长处理时间（README 第五章）。

        # 规则 1：风控优先于会员权益
        if customer["refund_count_90d"] > HIGH_RISK_REFUND_COUNT:
            return EligibilityResult(
                "不通过",
                f"客户近 90 天退款 {customer['refund_count_90d']} 次，"
                f"超过 {HIGH_RISK_REFUND_COUNT} 次阈值，属高风险账户，"
                "自动退款通道已关闭，需转人工客服审核（1-3 个工作日）",
            )

        # 规则 2：类目黑名单
        if order["category"] in NON_REFUNDABLE_CATEGORIES:
            return EligibilityResult(
                "不通过", f"商品类目「{order['category']}」不支持任何形式的退款"
            )

        days = order["signed_days_ago"]

        # 规则 3a：连最宽的窗口都超了，与退款理由无关
        if days > MAX_REFUND_WINDOW:
            return EligibilityResult(
                "不通过",
                f"订单已签收 {days} 天，已超出所有类型的退货窗口"
                f"（最宽为「质量问题」{MAX_REFUND_WINDOW} 天）",
            )

        # ── 第二段：以下判定确实取决于这两个参数，缺了才需要向用户确认 ──
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

        # 规则 3b：退货窗口
        window = 15 if reason_type == "质量问题" else (
            15 if customer["level"] == "金牌会员" else 7
        )
        if days > window:
            return EligibilityResult(
                "不通过",
                f"订单已签收 {days} 天，超出「{reason_type}」退货窗口"
                f"（{customer['level']} {window} 天）",
            )

        # 规则 4：商品条件（仅约束无理由退货）
        if reason_type != "质量问题" and item_condition != "未拆封":
            return EligibilityResult(
                "不通过",
                f"商品状态为「{item_condition}」，无理由退货要求未拆封、"
                "不影响二次销售",
            )

        return EligibilityResult(
            "通过",
            f"订单 {order_id} 符合「{reason_type}」退款条件"
            f"（签收 {days} 天 ≤ 窗口 {window} 天，商品状态：{item_condition or '不限'}）",
            refundable_amount=order["price"],
        )

    # ── 终局动作 ──────────────────────────────────────────────────────────
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
        """同一幂等键返回同一个单号，而不是重复落一笔（README 第七章）。"""
        if not idempotency_key:
            return None
        for row in eval_store.decision_log():
            if row["idempotency_key"] == idempotency_key:
                return RefundReceipt(row["receipt_no"], row["amount"])
        return None
