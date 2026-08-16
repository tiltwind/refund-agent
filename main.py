"""v1 演示入口 —— 客户档案与订单走 eval 数据源，政策检索走真实 Milvus。

    bash scripts/milvus.sh start       # 启 Milvus 2.5+，见 doc/rag/milvus-service.md
    python knowledge/seed_milvus.py    # 切片 doc/policy/ 并灌库（只需一次）
    python main.py                     # 需要 ANTHROPIC_API_KEY

想看检索链路每一步的中间产物（改写→路由→过滤→召回融合→重排→装配），
加 REFUND_AGENT_RAG_TRACE=on。

真正的服务入口是 app/main.py（FastAPI + 认证中间件，README 4.3），v1 还没写 ——
先用这个脚本把「查客户 → 查政策 → 判资格 → 落库 → 答复」这条链路跑通，
它和线上走的是**同一份工具代码、同一条代码路径**，区别只在入口注入的 context。
"""

from agent import registry
from app.context import RefundContext
from llm import chat
from services import eval_store, telemetry


def run_case(title: str, customer_id: str, request_id: str, message: str) -> None:
    print(f"\n{'=' * 70}\n场景：{title}\n用户（{customer_id}）：{message}\n{'-' * 70}")

    # 线上这个 context 由认证中间件从网关注入的 header 构造；这里由演示脚本
    # 直接构造。request_source 只能由服务端决定，绝不能来自请求体（README 9.3）。
    ctx = RefundContext(
        customer_id=customer_id,
        actor="self",
        request_id=request_id,
        # 演示脚本每个场景各跑一轮、互不相干，因此 session 与 request 一一对应；
        # 多轮对话下这里应该是会话 ID，同一通会话的每轮共用一个（README 8.2）。
        session_id=request_id,
        request_source="eval",
    )

    log_before = len(eval_store.decision_log())
    result = registry.get("v1").invoke(
        {"messages": [{"role": "user", "content": message}]},
        context=ctx,
        # 埋点走 config 而不是 context：context 是给工具层的业务身份，config 是给
        # LangGraph 运行时的回调通道，两者别混。没配 Langfuse 时这里是个空 dict。
        config=telemetry.trace_config(ctx, registry.meta("v1"), name=f"refund-chat:{title}"),
    )

    # 打印工具调用轨迹，观察五步 SOP 是否被完整执行
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"  ▶ {call['name']}({call['args']})")

    print(f"\nRefundAgent：{result['messages'][-1].text}")

    # 「说了」是否等于「做了」：答复里的单号必须来自真实落库的这一笔
    new_rows = eval_store.decision_log()[log_before:]
    if not new_rows:
        print("  ⚠️ 本轮没有任何终局动作落库")


def main() -> None:
    # 启动时先把「打哪个模型」「trace 报不报」两件事打出来：配错密钥、Langfuse 没起、
    # 模型名解析成了另一个供应商的，在这里一眼可见，不用等跑完三个场景才发现。
    print(chat.describe())
    print(telemetry.describe())

    # 场景一：金牌会员，签收 10 天。普通会员窗口 7 天会被拒，金牌 15 天 → 批准。
    run_case(
        "金牌会员的窗口期优待（预期：批准）",
        "C1001",
        "req-demo-001",
        "你好，订单 O2001 的耳机买回来一直没拆封，现在不想要了，想无理由退货。",
    )

    # 场景二：生鲜类目命中黑名单 → 拒绝。
    run_case(
        "不支持退款的类目（预期：拒绝）",
        "C1002",
        "req-demo-002",
        "O2002 那单三文鱼我想退掉，昨天才签收的。",
    )

    # 场景三：近 90 天退款 4 次，触发风控 → 拒绝自动退款、引导转人工。
    run_case(
        "高风险账户转人工（预期：拒绝并引导）",
        "C1003",
        "req-demo-003",
        "订单 O2003 的键盘有质量问题，按键失灵，要求退款！",
    )

    # 审计视角复盘所有终局动作。缺 actor 和 request_id 的流水事后追不到人、
    # 对不上链路（README 第七章）。
    print(f"\n{'=' * 70}\n决策流水\n{'-' * 70}")
    for row in eval_store.decision_log():
        print(
            f"  [{row['decision']}] 订单 {row['order_id']}  单号 {row['receipt_no']}  "
            f"¥{row['amount']:.2f}  actor={row['actor']}  "
            f"req={row['idempotency_key']}\n      {row['reason']}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        # 放 finally 里：链路中途抛异常时，那条失败的 trace 恰恰是最该看到的。
        telemetry.flush()
