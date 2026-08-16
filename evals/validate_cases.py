"""数据集自检 —— 把期望值喂给规则引擎，不自洽的用例直接拦下（2-design 6.2 第一层）。

    python evals/validate_cases.py                      # 默认校验 evals/dataset/d1
    python evals/validate_cases.py evals/dataset/d2

**零成本、不调模型、不连 Milvus**，所以它该在每次改规则引擎、每次加用例时先跑。
评估集自身也是被测系统的一部分：打分器写错、期望值口径滞后于规则变更，都会让报告给出
与事实相反的数字（2-design 6.1 里 ⑤ 那条反向箭头）。用例挂了先在这里确认一件事 ——
是 Agent 错了，还是用例的口径已经落后于规则引擎。

校验四类不一致：

1. **真值锚偏移**：`expected.eligibility.probe` 喂进规则引擎，返回的 verdict / 文案 /
   可退金额与用例写的期望对不上 —— 规则改了而用例没跟。
2. **结论自相矛盾**：`outcome=approved` 却期望 verdict=不通过、批准金额与可退金额不等、
   落库方向写反。
3. **工具断言与结论不符**：批准类用例没要求调 execute_refund，或没禁掉 record_refund_denial。
4. **数据引用悬空**：用例引用的客户不在 fixture 里（订单不存在是合法业务结论，不算）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import eval_store  # noqa: E402
from services.order.eval import EvalOrderService  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset" / "d1"

# outcome → (期望的规则引擎 verdict, 落库方向, 必须调用的终局工具, 必须禁用的终局工具)
OUTCOME_SPEC = {
    "approved": ("通过", "批准", "execute_refund", "record_refund_denial"),
    "denied": ("不通过", "拒绝", "record_refund_denial", "execute_refund"),
    "clarify": ("需补充", None, None, None),
    "ask_order_id": (None, None, None, None),
}

FINAL_TOOLS = ("execute_refund", "record_refund_denial")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checked = 0

    def fail(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")


def _load(path: Path) -> list[dict]:
    rows = []
    with (path / "cases.jsonl").open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"cases.jsonl 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return rows


def _check_turn(rep: Report, case: dict, idx: int, turn: dict, svc: EvalOrderService) -> None:
    where = f"{case['case_id']} turn{idx + 1}"
    exp = turn["expected"]
    outcome = exp["outcome"]

    if outcome not in OUTCOME_SPEC:
        rep.fail(where, f"未知 outcome「{outcome}」，可选：{list(OUTCOME_SPEC)}")
        return
    want_verdict, want_decision, must_tool, forbid_tool = OUTCOME_SPEC[outcome]

    # ── 1. 真值锚：期望值 vs 规则引擎 ────────────────────────────────
    elig = exp.get("eligibility")
    if want_verdict and not elig:
        rep.fail(where, f"outcome={outcome} 必须带 eligibility 真值锚")
    if elig:
        probe = elig["probe"]
        result = svc.check_eligibility(
            order_id=probe["order_id"],
            acting_user=case["context"]["customer_id"],
            reason_type=probe.get("reason_type", ""),
            item_condition=probe.get("item_condition", ""),
        )
        if result.verdict != elig["verdict"]:
            rep.fail(
                where,
                f"规则引擎判 {result.verdict!r}，用例期望 {elig['verdict']!r}"
                f"（probe={probe}）→ {result.reason}",
            )
        for frag in elig.get("reason_contains", []):
            if frag not in result.reason:
                rep.fail(where, f"规则引擎文案缺片段 {frag!r}，实际：{result.reason}")
        want_amount = elig.get("refundable_amount", 0.0)
        if abs(result.refundable_amount - want_amount) > 1e-6:
            rep.fail(
                where,
                f"可退金额不符：规则引擎 {result.refundable_amount}，用例 {want_amount}",
            )
        # 2. 结论自相矛盾
        if want_verdict and elig["verdict"] != want_verdict:
            rep.fail(where, f"outcome={outcome} 与 verdict={elig['verdict']!r} 矛盾")

    # ── 落库期望 ────────────────────────────────────────────────────
    log = exp.get("decision_log")
    if want_decision:
        if not log:
            rep.fail(where, f"outcome={outcome} 必须写明 decision_log")
        else:
            if log["decision"] != want_decision:
                rep.fail(where, f"落库方向写反：期望「{want_decision}」，用例写「{log['decision']}」")
            if elig and log["order_id"] != elig["probe"]["order_id"]:
                rep.fail(
                    where,
                    f"落库订单号 {log['order_id']} 与 probe {elig['probe']['order_id']} 不一致",
                )
            if outcome == "approved" and elig:
                if abs(log["amount"] - elig["refundable_amount"]) > 1e-6:
                    rep.fail(
                        where,
                        f"批准金额 {log['amount']} ≠ 规则引擎可退金额 "
                        f"{elig['refundable_amount']}",
                    )
            if outcome == "denied" and abs(log["amount"]) > 1e-6:
                rep.fail(where, f"拒绝流水的金额必须为 0，用例写了 {log['amount']}")
    elif log:
        rep.fail(where, f"outcome={outcome} 不应有终局动作，却写了 decision_log")

    # ── 3. 工具断言与结论是否自洽 ────────────────────────────────────
    tools = exp["tools"]
    must, forbid = tools.get("must_call", []), tools.get("must_not_call", [])
    if must_tool and must_tool not in must:
        rep.fail(where, f"outcome={outcome} 的 must_call 缺 {must_tool}")
    if forbid_tool and forbid_tool not in forbid:
        rep.fail(where, f"outcome={outcome} 的 must_not_call 缺 {forbid_tool}")
    if not want_decision:
        for name in FINAL_TOOLS:
            if name in must:
                rep.fail(where, f"outcome={outcome} 不该要求调用终局工具 {name}")
            if name not in forbid:
                rep.fail(where, f"outcome={outcome} 必须在 must_not_call 里禁掉 {name}")
    if set(must) & set(forbid):
        rep.fail(where, f"must_call 与 must_not_call 冲突：{sorted(set(must) & set(forbid))}")
    for name in tools.get("order", []):
        if name not in must:
            rep.fail(where, f"order 里的 {name} 没写进 must_call")

    # ── 答复断言 ────────────────────────────────────────────────────
    answer = exp["answer"]
    if answer["must_include_receipt_no"] != bool(want_decision):
        rep.fail(
            where,
            f"outcome={outcome} 的 must_include_receipt_no 应为 {bool(want_decision)}"
            "（有落库才有单号可引用）",
        )
    if set(answer["must_mention"]) & set(answer["must_not_mention"]):
        rep.fail(where, "must_mention 与 must_not_mention 有交集")

    rep.checked += 1


def validate(dataset: Path) -> Report:
    rep = Report()
    cases = _load(dataset)
    svc = EvalOrderService()

    seen_ids: set[str] = set()
    seen_req: dict[str, str] = {}

    for case in cases:
        cid = case["case_id"]
        if cid in seen_ids:
            rep.fail(cid, "case_id 重复")
        seen_ids.add(cid)

        req = case["context"]["request_id"]
        if req in seen_req:
            rep.fail(cid, f"request_id {req!r} 与 {seen_req[req]} 撞车 —— 幂等键必须唯一")
        seen_req[req] = cid

        if case["context"].get("request_source") != "eval":
            rep.fail(cid, "request_source 必须是 eval")

        # 4. 数据引用悬空：客户必须在 fixture 里（订单查不到是合法业务结论，不检查）
        customer_id = case["context"]["customer_id"]
        if customer_id not in eval_store.CUSTOMERS:
            rep.fail(cid, f"客户 {customer_id} 不在 evals/data/customers.json 里")
            continue

        # 每条用例一份独立数据副本：execute_refund 会改 refunded，跑批并发下会互相污染
        eval_store.begin_session()
        try:
            for i, turn in enumerate(case["turns"]):
                _check_turn(rep, case, i, turn, svc)
        finally:
            eval_store.end_session()

        if not case["turns"]:
            rep.fail(cid, "turns 为空")

    return rep


def main() -> None:
    dataset = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DATASET
    if not (dataset / "cases.jsonl").exists():
        raise SystemExit(f"找不到 {dataset / 'cases.jsonl'}")

    cases = _load(dataset)
    rep = validate(dataset)

    print(f"数据集 {dataset.name}：{len(cases)} 条用例 / {rep.checked} 轮对话通过校验")
    if rep.errors:
        print(f"\n✗ {len(rep.errors)} 处不自洽：")
        for err in rep.errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print("✓ 期望值与规则引擎口径一致")


if __name__ == "__main__":
    main()
