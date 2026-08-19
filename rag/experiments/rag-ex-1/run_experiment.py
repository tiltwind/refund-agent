"""实验 rag-ex-1：跑数据集 r1 的检索评测。

    python rag/experiments/rag-ex-1/run_experiment.py                    # 离线，全量 102 条
    python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001     # 只跑指定用例
    python rag/experiments/rag-ex-1/run_experiment.py --judge            # 加两个 LLM 指标
    python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)

被测对象是 [`search_policy`](rag/retrieving/milvus.py) 的六步链路，不经过 Agent。

两条路径，判分逻辑共用 `run_case`，落同一份 `result.json`：

| 路径 | 样本从哪来 | 上报 |
|---|---|---|
| 默认 | `rag/datasets/r1/cases.jsonl` | 无。三档 Recall 是要进门禁的数，不该被一个本地 Langfuse 实例的死活卡住 |
| `--langfuse` | Langfuse 数据集 `retrieval-cases-r1` | 分数写回 dataset run，六步中间产物随 trace 上报 |

默认只算三档 Recall 与两个辅助数（纯函数，不花模型钱，改写那一步除外）。加 `--judge`
才算 Context Recall 与 Context Relevance —— 那两个每条要多调两次 judge，慢一个量级，
而且是观察指标不进门禁（judge.py）。调参迭代跑默认路径，出报告时再带上 `--judge`。

前置：Milvus 起着并已灌库（`bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py`），
数据集自检过（`python rag/evals/validate_cases.py`）；`--judge` 要先拆好 claim
（`python rag/evals/generate_claims.py`）；`--langfuse` 还要先推数据集
（`python rag/evals/push_dataset.py`）。
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import scorers  # noqa: E402
from rag.evals.common import DATASET_DIR, load_env, read_cases  # noqa: E402

RESULT_PATH = HERE / "result.json"

RECALL = ("recall@1", "recall@3", "recall@10")
AUX = ("evidence_tokens", "duplicate_ratio")
OBS = ("context_recall", "context_relevance")
"""两个 LLM 指标。跟 RECALL 分开列是因为它们不进门禁：judge 有噪声，校准之前
只记录、看趋势（5-rag-eval 七）。"""


# ── 跑批 ──────────────────────────────────────────────────────────────────
class Progress:
    """并发跑批时的进度打印，整行拿锁打完再放。"""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.t0 = time.monotonic()
        self._lock = threading.Lock()

    def emit(self, row: dict) -> None:
        with self._lock:
            self.done += 1
            scores = row["scores"]
            if row["error"] and row["type"] != "unanswerable":
                note = row["error"]
            elif row["type"] == "unanswerable":
                # 「没抛异常」与「返回了证据」是两回事：实测 6 条里 5 条是重排全滤后
                # 拿到空列表 —— 兜底行为没生效，但也确实没给出不相关的条款，两者要分开报
                note = ("按预期抛异常" if scores.get("unanswerable_raised")
                        else f"未抛异常，{len(row.get('sections') or [])} 条证据")
            else:
                # `-` 是这一档不适用（multi_hop 的 recall@1），与判负分开显示
                marks = {1.0: "1", 0.0: "0", None: "-"}
                hit = "".join(marks[scores.get(name)] for name in RECALL)
                note = f"@1/@3/@10={hit} · {scores['evidence_tokens']} token"
                judged = "".join(
                    f" · {short}={scores[name]}"
                    for name, short in zip(OBS, ("CR", "CRel"))
                    if name in scores
                )
                note += judged
            print(
                f"  [{self.done:>{len(str(self.total))}}/{self.total}] "
                f"{row['case_id']} {row['elapsed_s']:.1f}s · {note}",
                flush=True,
            )

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def run_case(service, case: dict, judge_on: bool = False) -> dict:
    """跑一条用例并判分。`judge_on` 是 `--judge`，多算两个 LLM 指标。"""
    seeds = case["seed_chunk_id"]
    equivalent = case.get("acceptable_seed_chunk_ids")
    meta = case["meta"]
    row = {
        "case_id": case["case_id"],
        "query": case["query"],
        "type": case["type"],
        "style": case["style"],
        "doc_id": meta.get("doc_id", ""),
        "layer": meta.get("layer", ""),
        "kind": meta.get("kind", ""),
        "seed_chunk_id": seeds,
        "acceptable_seed_chunk_ids": equivalent,
        "error": None,
        "scores": {},
    }

    t0 = time.monotonic()
    try:
        top_k = 6 if case["type"] == "multi_hop" else None
        if top_k is None:
            sections, trace = service.search_with_trace(case["query"])
        else:
            sections, trace = service.search_with_trace(case["query"], top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        # 链路对「一条候选都没有」的口径是显式抛异常（milvus.py），unanswerable
        # 样本要的正是这个。其余类型抛异常就是执行失败，与判负分开统计。
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["outcome"] = "no_evidence" if type(exc).__name__ == "NoEvidenceError" else (
            "no_candidates" if type(exc).__name__ == "NoCandidatesError" else "error"
        )
        row["elapsed_s"] = round(time.monotonic() - t0, 1)
        if case["type"] == "unanswerable":
            row["scores"]["unanswerable_raised"] = 1.0
        elif row["outcome"] == "no_evidence":
            # 空证据是被测链路的结果，不是跑批执行失败；answerable 用例必须留在
            # Recall 分母里，并明确记为未命中。否则新增的安全兜底会让均值看起来变好。
            row["scores"] = {
                name: 0.0
                for name, k in (("recall@10", 10), ("recall@3", 3), ("recall@1", 1))
                if k >= len(seeds)
            }
            row["scores"].update({"evidence_tokens": 0, "duplicate_ratio": 0.0})
            row["sections"] = []
        return row
    row["elapsed_s"] = round(time.monotonic() - t0, 1)

    if case["type"] == "unanswerable":
        # 没有种子块，三档 Recall 的分母是 0 —— 算出来会是满分（空集是任何集合的
        # 子集），混进均值就是白送的分。这类样本只判兜底行为。
        row["scores"]["unanswerable_raised"] = 0.0
        row["sections"] = [s.section for s in sections]
        # 没抛异常时才有东西可判。Context Relevance 不要标注，正好用来量化 6.2 说的
        # 那件事：实测 6 条里 5 条被重排全滤成空证据，剩下一条返回了不相关的条款，
        # 它的相关度判出来是 0 —— 这个数正是「无适用条款」那条兜底判据的候选
        if judge_on:
            _judge_relevance(row, case, sections)
        return row

    # k < 种子块数的档次不计分：命中要求全中，两个种子块不可能同时排第 1，
    # `multi_hop` 的 recall@1 是结构性的 0。算进均值就等于按 multi_hop 的占比
    # 给 recall@1 加了一个固定折扣，改参数动不了它，版本间对比也读不出东西。
    row["scores"] = {
        name: scorers.recall_at_k(ids, seeds, k, equivalent)
        for name, ids, k in (
            ("recall@10", trace.candidate_ids, 10),
            ("recall@3", trace.evidence_ids, 3),
            ("recall@1", trace.evidence_ids, 1),
        )
        if k >= len(seeds)
    }
    row["scores"]["evidence_tokens"] = scorers.evidence_tokens(sections)
    row["scores"]["duplicate_ratio"] = scorers.duplicate_ratio(sections)
    # 掉分时的归因材料：种子块在候选里第几、在证据里第几，两个名次差就是重排的账
    row["seed_rank"] = {
        "candidate": scorers.seed_ranks(trace.candidate_ids, seeds),
        "evidence": scorers.seed_ranks(trace.evidence_ids, seeds),
    }
    row["rewrite_plan"] = trace.rewrite_plan
    row["rewrite_hash"] = trace.rewrite_hash
    row["sections"] = [s.section for s in sections]
    if judge_on:
        # claim 跟样本走（cases.jsonl 的 claims 字段），不在这里现拆：分母要在两次
        # run 之间保持一致，否则 Context Recall 的涨跌读不出是链路变了还是拆分变了
        _judge_recall(row, case.get("claims") or [], sections)
        _judge_relevance(row, case, sections)
    return row


def _judge_recall(row: dict, claims: list[str], sections) -> None:
    import judge

    result = judge.context_recall(claims, sections)
    _record(row, "context_recall", result)


def _judge_relevance(row: dict, case: dict, sections) -> None:
    import judge

    result = judge.context_relevance(case["query"], sections)
    _record(row, "context_relevance", result)


def _record(row: dict, name: str, result: dict) -> None:
    """判定结果落两处：分数进 `scores` 参与聚合，逐条理由进 `judge` 供翻查。

    judge 调用失败不写分数 —— 写个 0 会被均值当成「检索没召回」，把模型故障记到
    检索头上。失败的那条留在 `judge` 里，跑批末尾按数量汇报；「没有可判的东西」
    （空证据、没有 claim）走的是 `skipped`，不算故障。
    """
    row.setdefault("judge", {})[name] = result
    if result["score"] is not None:
        row["scores"][name] = result["score"]


# ── claim ────────────────────────────────────────────────────────────────
def check_claims(dataset: Path, wanted: set | None = None) -> int:
    """跑批前确认要判的样本都带 claim。

    缺了就退出，不是跳过那几条 —— 少几条样本的 Context Recall 仍然算得出一个数，
    看着正常，实际分母已经不是数据集那个了。补：`python rag/evals/generate_claims.py`。
    """
    from rag.evals.generate_claims import needs_claims

    cases = [
        c for c in read_cases(dataset)
        if needs_claims(c) and (not wanted or c["case_id"] in wanted)
    ]
    if missing := [c["case_id"] for c in cases if not c.get("claims")]:
        raise SystemExit(
            f"{len(missing)} 条样本还没有 claims：{missing[:5]}…\n"
            "跑 python rag/evals/generate_claims.py 补上"
        )
    return len(cases)


# ── Langfuse：dataset run ─────────────────────────────────────────────────
def case_of(item) -> dict:
    """Langfuse dataset item → 跑批用的 case，`push_dataset.py` 那套切分的反向。"""
    meta = dict(item.metadata or {})
    return {
        "case_id": item.id,
        "query": item.input["query"],
        "seed_chunk_id": item.expected_output["seed_chunk_id"],
        "acceptable_seed_chunk_ids": item.expected_output.get("acceptable_seed_chunk_ids") or None,
        "claims": item.expected_output.get("claims") or [],
        "type": meta.get("type", ""),
        "style": meta.get("style", ""),
        "meta": meta,
    }


def evaluate(*, output, **_):
    """把 `run_case` 已经算好的分搬成 Evaluation。

    判分不在这里做 —— 两条路径（离线、dataset run）必须给出同一个数字，打分逻辑
    只能有一份，就在 `run_case` 里。这个函数只负责往 Langfuse 上搬。
    """
    from langfuse import Evaluation

    failed = output["error"] and output["type"] != "unanswerable"
    out = [
        Evaluation(name=name, value=value, comment=_why(output, name))
        for name, value in output["scores"].items()
    ]
    out.append(
        Evaluation(name="run_error", value=1.0 if failed else 0.0, comment=output["error"] or "ok")
    )
    return out


def _why(row: dict, name: str) -> str:
    """判负时把归因材料带上 —— 光一个 0 在 UI 上说明不了改哪里。"""
    if name in OBS:
        result = (row.get("judge") or {}).get(name, {})
        if name == "context_recall":
            missed = [d["claim"] for d in result.get("detail", []) if not d["supported"]]
            return f"{result.get('hit')}/{result.get('n')} 条 claim 被支撑" + (
                "；没撑住：" + "｜".join(missed[:3]) if missed else ""
            )
        return f"{result.get('hit')}/{result.get('n')} 个内容单元相关；{result.get('note', '')}"
    if not name.startswith("recall"):
        return ""
    ranks = (row.get("seed_rank") or {}).get("candidate" if name == "recall@10" else "evidence", {})
    return "种子块名次 " + "、".join(f"{cid}={rank}" for cid, rank in ranks.items())


def aggregate(*, item_results, **_):
    from langfuse import Evaluation

    rows = [r.output for r in item_results if isinstance(r.output, dict)]
    scored = [row for row in rows if row["type"] != "unanswerable" and row.get("outcome", "success") != "error"]
    stats = summarize(scored)
    out = [
        Evaluation(name=name, value=stats[name], comment=f"n={stats['counted'][name]}")
        for name in RECALL
        if stats[name] is not None
    ]
    out += [
        Evaluation(name=name, value=stats[name])
        for name in AUX + OBS
        if stats[name] is not None
    ]
    return out


def run_dataset_run(args, service, wanted: set | None, judge_on: bool) -> tuple[list[dict], str | None, Progress]:
    """走 Langfuse 的 run_experiment：样本从 dataset 拉，分数写回 dataset run。

    样本不从本地 jsonl 读 —— dataset run 要挂在 Langfuse 那份数据集上，版本对比才有
    可比的对象。两份不一致时以推上去的那份为准，推之前跑 `validate_cases.py`。
    """
    from langfuse import get_client
    from services import telemetry

    if not telemetry.enabled():
        raise SystemExit("没配 LANGFUSE_PUBLIC_KEY / SECRET_KEY，dataset run 没处写")
    # 先走 telemetry 那条初始化路径再取 client：SDK 是进程内单例，两处各初始化一次
    # 会拿到不带 mask 钩子的那个，PII 就跟着 span 上去了（2-design 5.4）
    client = get_client()

    items = client.get_dataset(args.dataset_name).items
    if wanted:
        items = [item for item in items if item.id in wanted]
    if not items:
        raise SystemExit(f"数据集 {args.dataset_name} 里没有可跑的用例")

    progress = Progress(len(items))
    run_name = args.run_name or f"{args.dataset_name}-{len(items)}cases"
    print(f"→ {args.dataset_name}：{len(items)} 条 · 并发 {args.concurrency} · run={run_name}")

    def task(*, item, **_):
        row = run_case(service, case_of(item), judge_on)
        progress.emit(row)
        return row

    result = client.run_experiment(
        name=args.dataset_name,
        run_name=run_name,
        description="rag-ex-1 · 三档 Recall + 辅助数" + ("＋两个 LLM 指标" if judge_on else ""),
        data=items,
        task=task,
        evaluators=[evaluate],
        run_evaluators=[aggregate],
        max_concurrency=args.concurrency,
        metadata={key: str(value) for key, value in config(judge_on).items()},
    )
    client.flush()

    rows = []
    for item_result in result.item_results:
        row = dict(item_result.output)
        # trace_id 落盘：报告里指到某一条时，能直接跳到那条 trace 看六步
        row["trace_id"] = item_result.trace_id
        rows.append(row)
    return rows, result.dataset_run_url, progress


# ── 聚合 ──────────────────────────────────────────────────────────────────
def _fmt(value: float | None) -> str:
    return " —  " if value is None else f"{value:.3f}"


def _mean(rows: list[dict], name: str) -> float | None:
    values = [row["scores"][name] for row in rows if name in row["scores"]]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    out |= {name: _mean(rows, name) for name in RECALL}
    out |= {name: _mean(rows, name) for name in AUX}
    out |= {name: _mean(rows, name) for name in OBS}
    # 三档的分母不一样（recall@1 不含 multi_hop），不记下来就没法判断一档的涨跌
    # 是链路变了还是样本构成变了
    out["counted"] = {name: sum(1 for row in rows if name in row["scores"]) for name in RECALL}
    return out


def breakdown(rows: list[dict], key: str) -> dict:
    """按一个维度切均值。总均值只用来看趋势，能指向改哪个文件的是这些分档。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key] or "-"].append(row)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def config(judge_on: bool = False) -> dict:
    """被测参数的快照。

    两次 run 的分数只有在这些值相同的前提下才可比 —— 尤其是 collection：
    重新灌库而 chunk_id 漂了，Recall 会全线暴跌，看上去像检索退化。
    """
    from llm.embedding import bge_m3
    from llm.rerank import bge_reranker
    from rag.retrieving import milvus, store
    from rag.retrieving.pipeline import assemble, recall, rerank

    from rag.evals.common import judge_name, judge_reasoning, judge_structured

    return {
        # judge 换了模型，两个 LLM 指标的历史分数就不可比（5-rag-eval 七）。
        # 思考档位同理：关掉思考判出来的分与开着思考的不是一回事，实测一次判定
        # 99% 的输出 token 花在思考上，它不可能不影响判定结果
        "judge_model": judge_name() if judge_on else "off",
        "judge_reasoning": (judge_reasoning() or "模型默认") if judge_on else "off",
        "judge_structured": (judge_structured() or "默认") if judge_on else "off",
        "collection": store.COLLECTION,
        "embedding_model": bge_m3.MODEL_ID,
        "reranker": bge_reranker.MODEL_ID if bge_reranker.ENABLED else "off",
        "top_k": milvus.DEFAULT_TOP_K,
        "candidate_limit": recall.CANDIDATE_LIMIT,
        "rrf_k": recall.RRF_K,
        "min_score": rerank.MIN_SCORE,
        "relevance_weight": rerank.RELEVANCE_WEIGHT,
        "prior_weight": rerank.PRIOR_WEIGHT,
        "token_budget": assemble.TOKEN_BUDGET,
        "merge_max_parents": assemble.MERGE_MAX_PARENTS,
    }


def judge_errors(rows: list[dict]) -> list[str]:
    """judge 调用失败的清单。这些条目没写分数，均值的分母也就少一条 —— 不报出来的话
    一次网关抖动会静悄悄地把指标算在少一半样本上。"""
    return [
        f"{row['case_id']}/{name}: {result['error']}"
        for row in rows
        for name, result in (row.get("judge") or {}).items()
        if result.get("error")
    ]


def write_result(path: Path, *, dataset: str, run_name: str, run_url: str | None,
                 elapsed_s: float, rows: list[dict], judge_on: bool) -> None:
    scored = [row for row in rows if row["type"] != "unanswerable" and row.get("outcome", "success") != "error"]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    errors = [row for row in rows if row.get("outcome", "success") == "error"]
    outcomes = defaultdict(int)
    for row in rows:
        outcomes[row.get("outcome", "success" if not row["error"] else "error")] += 1

    payload = {
        "experiment": "rag-ex-1",
        "dataset": dataset,
        "run_name": run_name,
        "run_url": run_url,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config(judge_on),
        "cases_total": len(rows),
        "cases_scored": len(scored),
        "elapsed_s": round(elapsed_s, 1),
        "summary": summarize(scored)
        | {
            "unanswerable_raised": _mean(unanswerable, "unanswerable_raised"),
            "error_rate": round(len(errors) / len(rows), 3) if rows else 0.0,
            "outcomes": dict(sorted(outcomes.items())),
            # unanswerable 不进 Recall 均值，但它的 Context Relevance 有用：
            # 语料里没有的问题，检回的东西相关度应当明显低于正常样本（6.2）
            "unanswerable_relevance": _mean(unanswerable, "context_relevance"),
            "judge_errors": judge_errors(rows),
        },
        "breakdown": {key: breakdown(scored, key) for key in ("style", "type", "layer", "kind", "doc_id")},
        "cases": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_local(args, service, wanted: set | None, judge_on: bool) -> tuple[list[dict], None, Progress]:
    """离线跑批：样本从 jsonl 读，什么都不上报。三档 Recall 是要进门禁的数，
    不该被一个本地 Langfuse 实例的死活卡住。"""
    from concurrent.futures import ThreadPoolExecutor

    dataset = Path(args.dataset).resolve()
    cases = read_cases(dataset)
    if wanted:
        cases = [case for case in cases if case["case_id"] in wanted]
    if not cases:
        raise SystemExit(f"{dataset} 里没有可跑的用例")

    progress = Progress(len(cases))
    print(f"→ {dataset.name}：{len(cases)} 条 · 并发 {args.concurrency}")

    def task(case: dict) -> dict:
        row = run_case(service, case, judge_on)
        progress.emit(row)
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(task, cases))
    return rows, None, progress


# ── 入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="跑 r1 检索评测，算三档 Recall")
    parser.add_argument("dataset", nargs="?", default=str(DATASET_DIR), help="数据集目录（离线路径用）")
    parser.add_argument("--langfuse", action="store_true",
                        help="样本从 Langfuse 数据集拉，分数写回 dataset run，六步上报 trace")
    parser.add_argument("--dataset-name", default="retrieval-cases-r1",
                        help="Langfuse 上的数据集名，配合 --langfuse")
    parser.add_argument("--run-name", help="默认 <数据集>-<条数>cases；版本对比时传 git sha")
    parser.add_argument("--cases", nargs="*", help="只跑这些 case_id")
    parser.add_argument("--judge", action="store_true",
                        help="加算 Context Recall 与 Context Relevance，每条多两次 judge 调用；"
                             "先跑 python rag/evals/generate_claims.py 拆好 claim")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发用例数；嵌入和重排是本地模型，调太高只会互相抢算力")
    parser.add_argument("--out", help=f"指标落盘路径，默认 {RESULT_PATH.name}；"
                                      "只跑子集时默认不写，免得覆盖全量结果")
    args = parser.parse_args()

    load_env()

    # 离线跑批不上报检索 trace：那种 trace 不挂在任何 dataset run 上，堆在项目里只是
    # 噪音。必须在 import 检索链路之前设 —— SPAN 是模块常量，import 时就求值了。
    if not args.langfuse:
        os.environ["REFUND_AGENT_RAG_SPAN"] = "off"

    # claim 先查：缺了就该立刻退出，别等几个 G 的模型权重加载完再报错。
    # 只跑子集时只查子集 —— 调一条用例不该被另外 95 条挡住
    wanted = set(args.cases) if args.cases else None
    if args.judge:
        from rag.evals.common import judge_name

        n = check_claims(Path(args.dataset).resolve(), wanted)
        print(f"→ judge：{judge_name()}，{n} 条样本带 claim")

    from llm.embedding import embedder
    from llm.rerank import reranker
    from rag.retrieving.milvus import MilvusRagService

    # 先在主线程把两个本地模型加载起来：单例是 lru_cache，多线程同时首调会各自
    # 加载一份权重，几个 G 的内存白烧。
    print("正在加载模型 …")
    embedder()
    reranker()

    service = MilvusRagService()
    runner = run_dataset_run if args.langfuse else run_local
    rows, run_url, progress = runner(args, service, wanted, args.judge)

    scored = [row for row in rows if row["type"] != "unanswerable" and row.get("outcome", "success") != "error"]
    summary = summarize(scored)
    print(f"\n{'=' * 70}")
    print(f"  {len(scored)} 条计分 · 用时 {progress.elapsed():.1f}s")
    for name in RECALL:
        print(f"  {name:>16}: {_fmt(summary[name])}   n={summary['counted'][name]}")
    for name in AUX:
        print(f"  {name:>16}: {summary[name]}")
    for name in OBS:
        if summary[name] is not None:
            print(f"  {name:>16}: {_fmt(summary[name])}   n={sum(1 for r in scored if name in r['scores'])}")
    if failures := judge_errors(rows):
        print(f"\n  judge 失败 {len(failures)} 次（这些条目不计入均值）")
        for line in failures[:5]:
            print(f"    {line}")

    for key in ("style", "layer", "kind"):
        print(f"\n  按 {key} 分档")
        for value, stats in breakdown(scored, key).items():
            cells = "  ".join(f"{name}={_fmt(stats[name])}" for name in RECALL)
            print(f"    {value:<12} n={stats['n']:<4} {cells}")

    if run_url:
        print(f"\n  {run_url}")

    out = Path(args.out) if args.out else (None if args.cases else RESULT_PATH)
    if out:
        dataset = args.dataset_name if args.langfuse else str(Path(args.dataset).resolve().relative_to(ROOT))
        write_result(
            out,
            dataset=dataset,
            run_name=args.run_name or f"{Path(dataset).name}-{len(rows)}cases",
            run_url=run_url,
            elapsed_s=progress.elapsed(),
            rows=rows,
            judge_on=args.judge,
        )
        print(f"\n  指标已写入 {out}")


if __name__ == "__main__":
    main()
