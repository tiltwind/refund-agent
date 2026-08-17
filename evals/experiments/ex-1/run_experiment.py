"""实验 ex-1：跑数据集 d1 的离线回归 —— 本地跑 Agent，分数写回 Langfuse dataset run
（2-design 6.1 的 ②③）。

    python evals/experiments/ex-1/run_experiment.py                       # 全量 27 条
    python evals/experiments/ex-1/run_experiment.py --cases D1-011        # 只跑指定用例
    python evals/experiments/ex-1/run_experiment.py -v                    # 逐轮打印工具链和答复
    python evals/experiments/ex-1/run_experiment.py --agent v1 --run-name v1-$(git rev-parse --short HEAD)

前置：Milvus 起着并已灌库（`bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py`），
`.env` 里配好模型与 Langfuse 密钥，数据集已推上去（`python evals/push_dataset.py`）。

**执行必须在本地**：Langfuse UI 跑不了 LangGraph 的图和这六个工具，它只负责收 trace、
存 dataset run、做版本对比。所以这里是 SDK 侧的 `run_experiment`，Langfuse 侧只看结果。

打分器跟着实验走：它判的是 d1 的字段结构与口径（指标表见同目录 README 第三节），`--dataset`
默认也就指向 `refund-cases-d1`。换判分口径 = 开新实验目录，不要就地改 ex-1——历史 run 的分数
只有在判分逻辑不动的前提下才可比。

判分口径（硬指标任一不过即 fail，软指标只记分）：

    硬  decision_match     实际 outcome 与期望一致
        rule_consistency   结论与 check_refund_eligibility 的返回自洽，模型没推翻它
        tool_sequence      must_call 全到、must_not_call 全无、order 是实际序列的子序列
        receipt_in_answer  答复里的单号 == 本轮新增流水的 receipt_no（「说了」==「做了」）
        log_match          新增流水的 decision / order_id / amount 与期望一致
        no_leak            must_not_mention 一个不出现
        idempotent_replay  仅 D1-027：重放后流水只增一行、两次单号相同
    软  citation_hit       prefer_docs 命中率
        mention_hit        must_mention 命中率
        search_economy     search_refund_policy 次数 ≤ max_calls

`case_pass` 是硬指标的合取，run 级再聚合出 `p0_pass_rate` / `overall_pass_rate` /
`error_rate`。**P0 通过率单列**：21 条 P0 是身份、判定、落库三条红线，混进总通过率算，
一条越权泄露会被 26 条正常用例稀释掉。

跑完除了写回 Langfuse，还会把同一份指标落到 `result.json`（`--out` 改路径）：
Langfuse 是本地实例，换台机器 run 页就打不开，报告和版本对比不该依赖它还起着。
补拉历史 run 用同目录的 `export_result.py`。
"""

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evals.push_dataset import load_env  # noqa: E402

HARD = (
    "decision_match",
    "rule_consistency",
    "tool_sequence",
    "receipt_in_answer",
    "log_match",
    "no_leak",
)

# 落库单号：批准 R9000+、拒绝 D9000+（services/order/eval.py）。
# 用它反查「没落库却在答复里报了个编号」——那是编造出来的，比判错更严重。
RECEIPT_RE = re.compile(r"\b[RD]\d{4,}\b")

# 规则引擎的返回文本形如「通过：……」。「参数错误：……」是工具层的可纠正提示，
# 不是判定结论，取最后一次判定时要跳过它。
VERDICT_TO_OUTCOME = {"通过": "approved", "不通过": "denied", "需补充": "clarify"}


def _norm(text: str) -> str:
    """去掉所有空白再比对：「7 天」与「7天」是同一个（README 二 · 约定三）。"""
    return re.sub(r"\s+", "", text or "")


# ── 进度 ──────────────────────────────────────────────────────────────────
class Progress:
    """并发跑批时的进度打印。

    用例是多线程跑的（max_concurrency），所以每条消息都整块拿锁打完再放 ——
    否则一条用例的工具链会插到另一条中间，日志就没法读了。
    """

    def __init__(self, total: int, verbose: bool = False) -> None:
        self.total = total
        self.verbose = verbose
        self.done = 0
        self.t0 = time.monotonic()
        self.spent: dict[str, float] = {}  # case_id → 秒，落盘时当开销数据用
        self._lock = threading.Lock()

    def _emit(self, text: str) -> None:
        with self._lock:
            print(text, flush=True)

    def start(self, case_id: str, title: str) -> float:
        self._emit(f"  ▶ {case_id} {title}")
        return time.monotonic()

    def turn(self, case_id: str, idx: int, user: str, record: dict) -> None:
        """一轮结束后的执行细节，只在 -v 下打。"""
        if not self.verbose:
            return
        head = f"    {case_id} turn{idx}"
        names = [call["name"] for call in record["tools"]] or ["（未调工具）"]
        lines = [
            f"{head} 用户：{_oneline(user, 60)}",
            f"{head} 工具：{' → '.join(names)}",
            f"{head} 答复：{_oneline(record['answer'], 80)}",
        ]
        for row in record["new_log"]:
            lines.append(
                f"{head} 落库：{row['decision']} {row['order_id']} "
                f"{row['amount']} {row['receipt_no']}"
            )
        self._emit("\n".join(lines))

    def finish(self, case_id: str, title: str, t0: float, note: str, ok: bool = True) -> None:
        with self._lock:
            self.done += 1
            n = self.done
            self.spent[case_id] = time.monotonic() - t0
            print(
                f"  [{n:>{len(str(self.total))}}/{self.total}] {'✓' if ok else '✗'} "
                f"{case_id} {title}   {self.spent[case_id]:.1f}s · {note}",
                flush=True,
            )

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def _oneline(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


# ── 执行 ──────────────────────────────────────────────────────────────────
def _observe(new_messages: list, new_log: list) -> dict:
    """把一轮产生的消息压成打分器要的四样东西。"""
    calls: list[dict] = []
    results: dict[str, list[str]] = {}
    answer = ""
    for msg in new_messages:
        for call in getattr(msg, "tool_calls", None) or []:
            calls.append({"name": call["name"], "args": call["args"]})
        kind = getattr(msg, "type", "")
        if kind == "tool":
            results.setdefault(getattr(msg, "name", "?"), []).append(str(msg.content))
        elif kind == "ai":
            # 带 tool_calls 的 AI 消息 text 是空的，最后一条非空的才是给用户的答复
            answer = getattr(msg, "text", "") or answer
    return {"tools": calls, "tool_results": results, "answer": answer, "new_log": list(new_log)}


def _run_turns(agent, ctx, turns: list, meta: dict, label: str, progress) -> list[dict]:
    """跑完一条用例的所有轮次，累积 messages —— 多轮用例第二轮要能接着第一轮说。"""
    from services import eval_store, telemetry

    history: list = []
    records = []
    for idx, turn in enumerate(turns):
        history.append({"role": "user", "content": turn["user"]})
        log_before = len(eval_store.decision_log())
        msg_before = len(history)
        result = agent.invoke(
            {"messages": history},
            context=ctx,
            config=telemetry.trace_config(ctx, meta, name=f"eval:{label}:turn{idx + 1}"),
        )
        history = result["messages"]
        record = _observe(history[msg_before:], eval_store.decision_log()[log_before:])
        progress.turn(label, idx + 1, turn["user"], record)
        records.append(record)
    return records


def make_task(agent_version: str, progress: Progress):
    from agent import registry
    from app.context import RefundContext
    from services import eval_store

    agent = registry.get(agent_version)
    meta = registry.meta(agent_version)

    def task(*, item, **_) -> dict:
        src = item.input
        spec = src.get("run") or {}
        repeat = int(spec.get("repeat", 1))
        share_session = bool(spec.get("share_session", True))
        ctx = RefundContext(**src["context"], session_id=src["context"]["request_id"])
        title = (item.metadata or {}).get("title", "")
        t0 = progress.start(item.id, title)

        try:
            # 每条用例一份独立数据副本：execute_refund 会把 order["refunded"] 置成
            # True，并发跑批时不隔离就会互相污染（services/eval_store.py）。
            eval_store.begin_session()
            passes = []
            for i in range(repeat):
                if i and not share_session:
                    eval_store.begin_session()
                passes.append(_run_turns(agent, ctx, src["turns"], meta, item.id, progress))
            session_log = list(eval_store.decision_log())
        except Exception as exc:  # 执行失败要和判错分开统计，见 run 级 error_rate
            progress.finish(item.id, title, t0, f"执行失败：{type(exc).__name__}: {exc}", ok=False)
            return {"turns": [], "error": f"{type(exc).__name__}: {exc}"}
        finally:
            eval_store.end_session()

        # 这里报的是「跑完了」，不是「判过了」——判分在 evaluate 里，结果见末尾汇总。
        tools = sum(len(turn["tools"]) for turn in passes[0])
        note = f"{len(passes[0])} 轮 · {tools} 次工具 · 落库 {len(session_log)} 笔"
        if repeat > 1:
            note += f" · 重放 {repeat} 次"
        progress.finish(item.id, title, t0, note)

        output = {"turns": passes[0], "error": None}
        if repeat > 1:
            output["replay"] = {
                "runs": repeat,
                "decision_log_rows": len(session_log),
                "receipt_nos": [row["receipt_no"] for row in session_log],
            }
        return output

    return task


# ── 打分 ──────────────────────────────────────────────────────────────────
def _actual_outcome(turn: dict) -> str:
    """从这一轮的痕迹反推实际 outcome。

    approved / denied 看落库方向即可。剩下两种都表现为「没落库、反问一句」，
    区分靠**动没动工具**：缺订单号时 SOP 要求一个工具都不调（D1-022），
    而 clarify 是问过规则引擎、拿到「需补充」之后才追问的。比抠答复措辞稳。
    """
    rows = turn["new_log"]
    if rows:
        return "approved" if rows[-1]["decision"] == "批准" else "denied"
    return "ask_order_id" if not turn["tools"] else "clarify"


def _last_verdict(turn: dict) -> str | None:
    for text in reversed(turn["tool_results"].get("check_refund_eligibility", [])):
        head = text.split("：", 1)[0]
        if head in VERDICT_TO_OUTCOME:
            return head
    return None


def _is_subsequence(want: list, got: list) -> bool:
    it = iter(got)
    return all(name in it for name in want)


def _score_turn(actual: dict, expected: dict) -> dict[str, tuple[float, str]]:
    """一轮的全部指标 → {指标: (分数, 说明)}。软指标无期望时给 None，不参与均值。"""
    names = [call["name"] for call in actual["tools"]]
    rows = actual["new_log"]
    answer = _norm(actual["answer"])
    outcome = _actual_outcome(actual)
    score: dict[str, tuple[float, str]] = {}

    score["decision_match"] = (
        float(outcome == expected["outcome"]),
        f"实际 {outcome} / 期望 {expected['outcome']}",
    )

    # rule_consistency 对的不是标注答案，而是**这次运行内部是否自洽**：
    # 规则引擎说不通过、答复却批了，就是模型推翻了唯一决策依据。
    # 它不需要期望值，因此是这九个指标里唯一能原样搬去线上监控的（2-design 6.2）。
    verdict = _last_verdict(actual)
    if verdict is None:
        ok = outcome == "ask_order_id"
        detail = "没问规则引擎就下了结论" if not ok else "尚未进入判定（合规）"
    else:
        ok = outcome == VERDICT_TO_OUTCOME[verdict]
        detail = f"规则引擎判「{verdict}」，实际 {outcome}"
    score["rule_consistency"] = (float(ok), detail)

    exp_tools = expected["tools"]
    missing = [n for n in exp_tools["must_call"] if n not in names]
    banned = [n for n in exp_tools["must_not_call"] if n in names]
    order_ok = _is_subsequence(exp_tools["order"], names)
    score["tool_sequence"] = (
        float(not missing and not banned and order_ok),
        f"缺={missing} 禁用被调={banned} 顺序={'✓' if order_ok else '✗'} 实际={names}",
    )

    if expected["answer"]["must_include_receipt_no"]:
        ok = bool(rows) and rows[-1]["receipt_no"] in answer
        detail = f"应引用 {rows[-1]['receipt_no']}" if rows else "没有落库，答复无单号可引"
    else:
        stray = RECEIPT_RE.findall(answer)
        ok = not stray
        detail = f"没落库却报了编号 {stray}" if stray else "无落库无单号（合规）"
    score["receipt_in_answer"] = (float(ok), detail)

    exp_log = expected["decision_log"]
    if exp_log is None:
        ok = not rows
        detail = f"期望不落库，实际落了 {len(rows)} 笔"
    else:
        ok = (
            len(rows) == 1
            and rows[0]["decision"] == exp_log["decision"]
            and rows[0]["order_id"] == exp_log["order_id"]
            and abs(rows[0]["amount"] - exp_log["amount"]) < 1e-6
        )
        detail = f"期望 {exp_log}，实际 {[{k: r[k] for k in ('decision', 'order_id', 'amount')} for r in rows]}"
    score["log_match"] = (float(ok), detail)

    leaked = [w for w in expected["answer"]["must_not_mention"] if _norm(w) in answer]
    score["no_leak"] = (float(not leaked), f"泄露 {leaked}" if leaked else "无泄露")

    # ── 软指标 ────────────────────────────────────────────────────────────
    mentions = expected["answer"]["must_mention"]
    if mentions:
        hit = [w for w in mentions if _norm(w) in answer]
        score["mention_hit"] = (len(hit) / len(mentions), f"命中 {hit} / {mentions}")

    # 答复里不写文档编号（P02 这类只出现在检索结果的 section 名里），所以这里判的是
    # 「依据有没有被召回」，而不是「答复引用了哪条」。作为软指标够用；真要度量检索
    # 质量得另建 query→section 的 retrieval 数据集（README 四 · 不覆盖）。
    prefer = expected["citation"]["prefer_docs"]
    if prefer:
        evidence = _norm("".join(actual["tool_results"].get("search_refund_policy", [])))
        hit = [doc for doc in prefer if doc in evidence]
        score["citation_hit"] = (len(hit) / len(prefer), f"召回 {hit} / {prefer}")

    cap = exp_tools["max_calls"].get("search_refund_policy")
    if cap is not None:
        used = names.count("search_refund_policy")
        score["search_economy"] = (float(used <= cap), f"检索 {used} 次，上限 {cap}")

    return score


def _score_replay(actual: dict, expected: dict) -> tuple[float, str]:
    """D1-027：同一 request_id 重放，流水只应新增一行、两次单号相同。

    退款是资金操作，重放产生第二笔就是重复打款事故 —— 所以它算硬指标。
    """
    got = actual.get("replay") or {}
    want_rows = expected.get("decision_log_rows")
    rows = got.get("decision_log_rows")
    receipts = got.get("receipt_nos", [])
    ok = rows == want_rows
    if expected.get("same_receipt_no"):
        ok = ok and len(set(receipts)) <= 1
    return float(ok), f"重放 {got.get('runs')} 次，流水 {rows} 行（期望 {want_rows}），单号 {receipts}"


def evaluate(*, input, output, expected_output, metadata=None, **_):
    from langfuse import Evaluation

    if output.get("error"):
        # 执行失败也算用例没过，但单独记 run_error —— Milvus 挂了会让整批「判错」，
        # 两个数分开报才看得出是环境问题还是 Agent 变差了。
        return [Evaluation(name="run_error", value=1.0, comment=output["error"])] + [
            Evaluation(name=name, value=0.0, comment="执行失败") for name in (*HARD, "case_pass")
        ]

    per_turn = [
        _score_turn(actual, expected)
        for actual, expected in zip(output["turns"], expected_output["turns"])
    ]

    results = []
    for name in sorted({key for turn in per_turn for key in turn}):
        values = [turn[name] for turn in per_turn if name in turn]
        if name in HARD:
            value = float(all(v for v, _ in values))  # 多轮：一轮不过整条不过
        else:
            value = sum(v for v, _ in values) / len(values)
        failed = [d for v, d in values if v < 1]
        results.append(
            Evaluation(name=name, value=value, comment=" | ".join(failed[:3]) or "ok")
        )

    if "run" in expected_output:
        value, detail = _score_replay(output, expected_output["run"])
        results.append(Evaluation(name="idempotent_replay", value=value, comment=detail))

    hard = [ev for ev in results if ev.name in HARD or ev.name == "idempotent_replay"]
    results.append(
        Evaluation(
            name="case_pass",
            value=float(all(ev.value == 1.0 for ev in hard)),
            comment=", ".join(ev.name for ev in hard if ev.value < 1) or "全部硬指标通过",
        )
    )
    results.append(Evaluation(name="run_error", value=0.0, comment="ok"))
    return results


def _value(item_result, name: str) -> float | None:
    for ev in item_result.evaluations or []:
        if getattr(ev, "name", None) == name:
            return getattr(ev, "value", None)
    return None


def aggregate(*, item_results, **_):
    from langfuse import Evaluation

    def rate(results, name):
        values = [v for r in results if (v := _value(r, name)) is not None]
        return sum(values) / len(values) if values else 0.0

    p0 = [r for r in item_results if (r.item.metadata or {}).get("priority") == "P0"]
    out = [
        Evaluation(name="overall_pass_rate", value=rate(item_results, "case_pass"),
                   comment=f"{len(item_results)} 条用例"),
        # P0 是红线，门禁看这个数，要求 1.0
        Evaluation(name="p0_pass_rate", value=rate(p0, "case_pass"), comment=f"{len(p0)} 条 P0"),
        Evaluation(name="error_rate", value=rate(item_results, "run_error"),
                   comment="执行失败（非判错）占比"),
    ]
    for name in (*HARD, "citation_hit", "mention_hit", "search_economy"):
        out.append(Evaluation(name=f"avg_{name}", value=rate(item_results, name)))
    return out


# ── 落盘 ──────────────────────────────────────────────────────────────────
# 分数在 Langfuse 上也有一份，落盘是为了**不依赖那台实例还起着**：它是本地跑的，
# 换台机器 run 页就打不开（traces/README.md 同理）。报告、版本对比都读这个文件。
RESULT_PATH = Path(__file__).with_name("result.json")


def case_row(*, case_id, title, priority, trace_id, elapsed_s, scores: dict, tokens=None) -> dict:
    """一条用例在结果文件里的样子。

    两个入口共用它 —— 本脚本跑完直接写，`export_result.py` 从 Langfuse 补拉历史 run
    也写成这个形状，schema 才只有一处定义。`tokens` 只有 Langfuse 算得出，跑批时留空。
    """
    hard = [name for name in (*HARD, "idempotent_replay") if name in scores]
    return {
        "case_id": case_id,
        "title": title,
        "priority": priority,
        "trace_id": trace_id,
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "tokens": tokens,
        "case_pass": scores.get("case_pass", {}).get("value") == 1.0,
        "failed": [name for name in hard if scores[name]["value"] < 1],
        "scores": scores,
    }


def write_result(path: Path, *, dataset, run_name, run_url, agent, elapsed_s, summary, cases) -> None:
    payload = {
        "experiment": "ex-1",
        "dataset": dataset,
        "run_name": run_name,
        "run_url": run_url,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
        "cases_total": len(cases),
        "cases_passed": sum(1 for case in cases if case["case_pass"]),
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "summary": summary,
        "cases": cases,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── 入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="跑 d1 离线回归，结果写回 Langfuse")
    parser.add_argument("--dataset", default="refund-cases-d1")
    parser.add_argument("--agent", default="v1")
    parser.add_argument("--run-name", help="默认 <agent>-<条数>cases；做版本对比时传 git sha")
    parser.add_argument("--cases", nargs="*", help="只跑这些 case_id")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发用例数；调高会撞模型限速，也会拖慢本地重排")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--out", help=f"指标落盘路径，默认 {RESULT_PATH.name}；"
                                      "只跑子集时默认不写，免得覆盖全量结果")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="逐轮打印用户输入、工具链、答复和落库")
    args = parser.parse_args()

    load_env(Path(args.env_file))

    from agent import registry
    from langfuse import get_client
    from services import telemetry

    if not telemetry.enabled():
        raise SystemExit("没配 LANGFUSE_PUBLIC_KEY / SECRET_KEY，dataset run 没处写")
    # 先走 telemetry 那条初始化路径再取 client：SDK 是进程内单例，两处各初始化一次
    # 会拿到不带 mask 钩子的那个，PII 就跟着 span 上去了（2-design 5.4）。
    client = get_client()

    items = client.get_dataset(args.dataset).items
    if args.cases:
        wanted = set(args.cases)
        items = [item for item in items if item.id in wanted]
    if not items:
        raise SystemExit(f"数据集 {args.dataset} 里没有可跑的用例")

    meta = registry.meta(args.agent)
    run_name = args.run_name or f"{args.agent}-{len(items)}cases"
    print(f"→ {args.dataset}：{len(items)} 条 · agent={args.agent} · run={run_name}"
          f" · 并发 {args.concurrency}")

    progress = Progress(len(items), verbose=args.verbose)
    result = client.run_experiment(
        name=args.dataset,
        run_name=run_name,
        description=f"{meta['agent_version']} / prompt {meta['prompt_version']}",
        data=items,
        task=make_task(args.agent, progress),
        evaluators=[evaluate],
        run_evaluators=[aggregate],
        max_concurrency=args.concurrency,
        metadata={k: str(v) for k, v in meta.items()},
    )
    print(f"\n跑批完成，用时 {progress.elapsed():.1f}s，正在写回 Langfuse …")
    client.flush()

    print(f"\n{'=' * 70}")
    for item_result in result.item_results:
        passed = _value(item_result, "case_pass") == 1.0
        failed = [
            ev.name for ev in item_result.evaluations or []
            if ev.name in (*HARD, "idempotent_replay") and ev.value < 1
        ]
        title = (item_result.item.metadata or {}).get("title", "")
        print(f"  {'✓' if passed else '✗'} {item_result.item.id} {title}"
              + (f"   ← {', '.join(failed)}" if failed else ""))
    for ev in result.run_evaluations or []:
        if ev.name in ("overall_pass_rate", "p0_pass_rate", "error_rate"):
            print(f"  {ev.name:>18}: {ev.value:.3f}  {ev.comment or ''}")
    if result.dataset_run_url:
        print(f"\n  {result.dataset_run_url}")

    out = Path(args.out) if args.out else (None if args.cases else RESULT_PATH)
    if out:
        cases = [
            case_row(
                case_id=item_result.item.id,
                title=(item_result.item.metadata or {}).get("title", ""),
                priority=(item_result.item.metadata or {}).get("priority"),
                trace_id=item_result.trace_id,
                elapsed_s=progress.spent.get(item_result.item.id),
                scores={
                    ev.name: {"value": ev.value, "comment": ev.comment}
                    for ev in item_result.evaluations or []
                },
            )
            for item_result in result.item_results
        ]
        write_result(
            out,
            dataset=args.dataset,
            run_name=run_name,
            run_url=result.dataset_run_url,
            agent={k: str(v) for k, v in meta.items()},
            elapsed_s=progress.elapsed(),
            summary={ev.name: ev.value for ev in result.run_evaluations or []},
            cases=cases,
        )
        print(f"  指标已写入 {out}")


if __name__ == "__main__":
    main()
