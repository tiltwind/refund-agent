"""发送 5 条模拟线上流量，用于验证 Langfuse 线上监控。

    python prod/main.py

脚本使用 prod 业务数据源，并把 trace 上报到 ``production`` 环境。
"""

import os
import sys
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.push_dataset import load_env  # noqa: E402


load_env(ROOT / ".env")
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "production"

from agent import registry  # noqa: E402
from app.context import RefundContext  # noqa: E402
from llm import chat  # noqa: E402
from services import online_monitor, prod_store, telemetry  # noqa: E402


SCENARIOS = [
    (
        "金牌会员窗口期内退款",
        "C1001",
        "订单 O2001 的耳机一直没拆封，现在不想要了，想申请无理由退货。",
    ),
    (
        "生鲜类目退款",
        "C1002",
        "O2002 那单三文鱼收到后不新鲜，我要退款。",
    ),
    (
        "高风险账户退款",
        "C1003",
        "订单 O2003 的机械键盘有质量问题，好几个键都失灵了，要求退款。",
    ),
    (
        "缺少退款原因",
        "C1004",
        "订单 O2004 我想退掉。",
    ),
    (
        "缺少订单号",
        "C1004",
        "你好，我要申请退款。",
    ),
]


def send_request(title: str, customer_id: str, message: str) -> None:
    request_id = f"sim-prod-{uuid4().hex}"
    ctx = RefundContext(
        customer_id=customer_id,
        actor="self",
        request_id=request_id,
        session_id=request_id,
        request_source="prod",
    )

    log_before = len(prod_store.decision_log())
    with telemetry.trace_turn(ctx, registry.meta("v1"), message) as trace:
        result = registry.get("v1").invoke(
            {"messages": [{"role": "user", "content": message}]},
            context=ctx,
            config=trace.config,
        )
        turn = trace.finish(
            result["messages"][1:],
            prod_store.decision_log()[log_before:],
        )

    outcome = online_monitor.actual_outcome(turn)
    print(f"[{outcome}] {title}  request_id={request_id}")
    print(f"  用户：{message}")
    print(f"  Agent：{turn['answer']}\n")


def main() -> None:
    prod_store.initialize()
    print(f"SQLite: {prod_store.database_path()}")
    print(chat.describe())
    print(telemetry.describe())
    print("发送 5 条模拟线上请求\n")
    for scenario in SCENARIOS:
        send_request(*scenario)


if __name__ == "__main__":
    try:
        main()
    finally:
        telemetry.flush()
