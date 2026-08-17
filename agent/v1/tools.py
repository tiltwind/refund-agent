"""v1 工具层 —— schema ↔ 业务动作的双向翻译（2-design 第二章）。

这一层**只做三件事**：校验模型填的参数、调 services/ 拿结果、把结果渲染成
模型好读的文本。它不含业务规则（在规则服务）、不做授权判定（在下游）、
也不关心 HTTP / gRPC 协议细节（在 services/）。

`runtime: ToolRuntime[RefundContext]` 不会出现在发给模型的 tool schema 里 ——
customer_id 和 request_id 由此进入工具，模型看不见也改不了。
"""

from langchain.tools import ToolRuntime, tool

from app.context import RefundContext
from rag.retrieving.protocol import PolicySection
from services.customer.protocol import CustomerProfile
from services.factory import customer_service, order_service, rag_service, rule_service

# 规则引擎认的取值 —— 与 docstring 保持一致，但**由代码强制**。
# docstring 对模型只是建议：模型给 item_condition 填过「未使用」，引擎当成
# 「非未拆封」照常判定，结论碰巧对了；换个订单就会把本该通过的申请拒掉。
REASON_TYPES = ("无理由", "质量问题")
ITEM_CONDITIONS = ("未拆封", "已拆封", "已使用")


# ── 渲染 ──────────────────────────────────────────────────────────────────
def _render_profile(p: CustomerProfile) -> str:
    orders = [
        f"  - {o.order_id}：{o.product}（{o.category}），实付 ¥{o.price:.2f}，"
        f"签收于 {o.signed_days_ago} 天前" + ("，已退款" if o.refunded else "")
        for o in p.orders
    ]
    return (
        f"客户 {p.customer_id}（{p.name}）\n"
        f"会员等级：{p.level}\n"
        f"注册时间：{p.register_date}\n"
        f"近 90 天退款次数：{p.refund_count_90d}\n"
        f"名下订单：\n" + ("\n".join(orders) or "  （无）")
    )


def _render_policies(sections: list[PolicySection]) -> str:
    """渲染成「内容 + 来源 + 时间 + 层级 + 相关性理由」。

    后四项不是给模型看的装饰，各有下游用途：来源用于答复引用与事后审计；
    生效日期让模型知道这一版还算不算数；层级提示它答复消费者该引平台条款、
    法规只用于判断平台条款是否有效；相关性理由与分数是坏 case 出现时区分
    「检索错了」和「模型答错了」的唯一抓手。
    """
    if not sections:
        return "未检索到相关政策条款"
    blocks = []
    for i, s in enumerate(sections, 1):
        layer = "平台条款（答复消费者的直接依据）" if s.layer == "platform" else "法律法规（法定底线）"
        blocks.append(
            f"[E{i}] {s.section}\n"
            f"  来源: {s.source_path}  |  生效: {s.effective_date} 起  |  {layer}\n"
            f"  相关性: {s.reason}\n"
            f"  ---\n"
            f"{s.text}"
        )
    return "\n\n".join(blocks)


# ── 工具 ──────────────────────────────────────────────────────────────────
@tool
def get_customer_info(runtime: ToolRuntime[RefundContext]) -> str:
    """查询**当前客户**的档案：会员等级、近 90 天退款次数、名下订单。
    处理退款请求时第一步调用。客户身份由系统自动带入，无需也无法指定。"""
    ctx = runtime.context
    profile = customer_service(ctx).get_profile(ctx.customer_id)
    return _render_profile(profile)


@tool
def search_refund_policy(query: str, runtime: ToolRuntime[RefundContext]) -> str:
    """检索退款政策文档。判定退款资格前必须调用一次。

    query 写成**一个完整的问句**，把商品类目、签收天数、会员等级、商品状态、
    用户诉求都写进这个句子（例：「金牌会员签收 10 天的耳机未拆封，无理由退货
    还在窗口期内吗？」），不要写成关键词堆叠 —— 检索器判断的是「哪条条款回答
    了这个问题」，没有问句就只能退化成算主题相似度，会把措辞相近但答非所问的
    条款排到前面。

    一次即可拿到相关条款；只有返回的条款不足以覆盖判定要素时才需要再查。"""
    sections = rag_service(runtime.context).search_policy(query)
    return _render_policies(sections)


@tool
def check_refund_eligibility(
    order_id: str,
    runtime: ToolRuntime[RefundContext],
    reason_type: str = "",
    item_condition: str = "",
) -> str:
    """判定订单退款资格。reason_type 取值：无理由 / 质量问题；
    item_condition 取值：未拆封 / 已拆封 / 已使用。

    这两个参数可以留空：若订单已命中硬否决规则（不存在、已退款、类目黑名单、
    高风险账户、超出所有退货窗口），无需它们即可定案。确实需要补充时，
    本工具会返回「需补充：……」明确告诉你该向用户确认什么。

    必须先查询客户信息和政策文档，再调用本工具。判定结果是最终依据。"""
    # 返回**可纠正的错误提示**而不是抛异常 —— 模型看到提示能自行改正重试，
    # 比整轮失败好。
    if reason_type and reason_type not in REASON_TYPES:
        return (
            f"参数错误：reason_type 只能是「无理由」或「质量问题」，"
            f"收到「{reason_type}」。请按用户的实际诉求重新判断后再次调用。"
        )
    if item_condition and item_condition not in ITEM_CONDITIONS:
        return (
            f"参数错误：item_condition 只能是「未拆封」「已拆封」「已使用」，"
            f"收到「{item_condition}」。请按用户描述的商品状态重新判断后再次调用。"
        )

    ctx = runtime.context
    result = rule_service(ctx).check_eligibility(
        order_id=order_id,
        acting_user=ctx.customer_id,
        reason_type=reason_type,
        item_condition=item_condition,
    )
    text = f"{result.verdict}：{result.reason}"
    if result.verdict == "通过":
        text += f"，可退金额 ¥{result.refundable_amount:.2f}"
    return text


@tool
def execute_refund(
    order_id: str, amount: float, reason: str, runtime: ToolRuntime[RefundContext]
) -> str:
    """批准退款并发起打款。仅在 check_refund_eligibility 返回「通过」后调用，
    amount 使用其给出的可退金额，reason 简述批准依据。
    返回的退款单号必须写进给用户的答复里。"""
    ctx = runtime.context
    try:
        receipt = order_service(ctx).execute_refund(
            order_id=order_id,
            acting_user=ctx.customer_id,
            amount=amount,
            reason=reason,
            idempotency_key=ctx.request_id,  # 同键不重复打款（2-design 第四章）
        )
    except ValueError as exc:
        return f"操作失败：{exc}"
    return (
        f"退款已批准，退款单号 {receipt.receipt_no}，"
        f"¥{receipt.amount:.2f} 将在 1-3 个工作日原路退回"
    )


@tool
def record_refund_denial(
    order_id: str, reason: str, runtime: ToolRuntime[RefundContext]
) -> str:
    """拒绝退款申请并落库。仅在 check_refund_eligibility 返回「不通过」后调用，
    reason 需引用具体政策依据，便于向用户解释。
    返回的受理编号必须写进给用户的答复里。"""
    # 拒绝也要给一个可引用的编号：一来用户凭它查询和申诉，二来这个编号只有
    # 真的调用了本工具才拿得到 —— 答复里必须引用它，就把「说了」和「做了」
    # 绑在了一起（1-architecture 第二章）。
    ctx = runtime.context
    receipt = order_service(ctx).record_denial(
        order_id=order_id,
        acting_user=ctx.customer_id,
        reason=reason,
        idempotency_key=ctx.request_id,
    )
    return f"已记录拒绝决策，受理编号 {receipt.receipt_no}：订单 {order_id}，原因：{reason}"


TOOLS = [
    get_customer_info,
    search_refund_policy,
    check_refund_eligibility,
    execute_refund,
    record_refund_denial,
]
