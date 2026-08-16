"""eval 数据源的加载与会话隔离 —— 只被 services/*/eval.py 使用。

只覆盖客户档案与订单两类数据。**政策条款不在这里**：检索无论 prod 还是 eval
都直连 Milvus（2-design 3.4），语料就是 doc/policy/ 下的政策文档本身，
由 knowledge/seed_milvus.py 切片后写入。

两件事：

1. **加载** `evals/data/*.json`，除 Milvus 外不连任何线上服务（2-design 6.3）。
2. **会话隔离**：评估跑批要并发，而 execute_refund 会把 order["refunded"]
   置成 True —— 两条用例同时跑就会互相污染。用 contextvars 给每条用例发一份
   独立副本，各改各的，互不可见。没开会话时退回模块级对象，单次演示最直观。
"""

import contextvars
import copy
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "evals" / "data"


def _load(name: str) -> dict:
    with (_DATA_DIR / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)


# `_note` 是给人读的边界标注（2-design 3.3），不参与任何判定，加载时剔除
CUSTOMERS: dict[str, dict] = {
    cid: {k: v for k, v in row.items() if k != "_note"}
    for cid, row in _load("customers").items()
}
ORDERS: dict[str, dict] = {
    oid: {k: v for k, v in row.items() if k != "_note"}
    for oid, row in _load("orders").items()
}

# 终局动作的落库目标：批准 / 拒绝各留一条流水
DECISION_LOG: list[dict] = []

_session: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "refund_eval_session", default=None
)


def begin_session() -> dict:
    """为当前上下文开一份独立的数据副本（幂等：重复调用会重新拷贝）。"""
    session = {
        "customers": copy.deepcopy(CUSTOMERS),
        "orders": copy.deepcopy(ORDERS),
        "decision_log": [],
    }
    _session.set(session)
    return session


def end_session() -> None:
    _session.set(None)


def customers() -> dict[str, dict]:
    session = _session.get()
    return session["customers"] if session else CUSTOMERS


def orders() -> dict[str, dict]:
    session = _session.get()
    return session["orders"] if session else ORDERS


def decision_log() -> list[dict]:
    session = _session.get()
    return session["decision_log"] if session else DECISION_LOG
