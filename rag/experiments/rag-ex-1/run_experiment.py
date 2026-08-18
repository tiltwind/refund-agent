"""实验 rag-ex-1：跑数据集 r1 的检索评测。

    python rag/experiments/rag-ex-1/run_experiment.py                    # 离线，全量 102 条
    python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001     # 只跑指定用例
    python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)

被测对象是 [`search_policy`](rag/retrieving/milvus.py) 的六步链路，不经过 Agent。

两条路径，判分逻辑共用 `run_case`，落同一份 `result.json`：

| 路径 | 样本从哪来 | 上报 |
|---|---|---|
| 默认 | `rag/datasets/r1/cases.jsonl` | 无。三档 Recall 是要进门禁的数，不该被一个本地 Langfuse 实例的死活卡住 |
| `--langfuse` | Langfuse 数据集 `retrieval-cases-r1` | 分数写回 dataset run，六步中间产物随 trace 上报 |

这一版只算三档 Recall 与两个辅助数，不调 judge，因此不花模型钱（改写那一步除外）。

前置：Milvus 起着并已灌库（`bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py`），
数据集自检过（`python rag/evals/validate_cases.py`）；`--langfuse` 还要先推数据集
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
                note = "按预期抛异常" if scores.get("unanswerable_raised") else "未抛异常，返回了证据"
            else:
                # `-` 是这一档不适用（multi_hop 的 recall@1），与判负分开显示
                marks = {1.0: "1", 0.0: "0", None: "-"}
                hit = "".join(marks[scores.get(name)] for name in RECALL)
                note = f"@1/@3/@10={hit} · {scores['evidence_tokens']} token"
            print(
                f"  [{self.done:>{len(str(self.total))}}/{self.total}] "
                f"{row['case_id']} {row['elapsed_s']:.1f}s · {note}",
                flush=True,
            )

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def run_case(service, case: dict) -> dict:
    seeds = case["seed_chunk_id"]
    meta = case["meta"]
    row = {
        "case_id": case["case_id"],
        "type": case["type"],
        "style": case["style"],
        "doc_id": meta.get("doc_id", ""),
        "layer": meta.get("layer", ""),
        "kind": meta.get("kind", ""),
        "seed_chunk_id": seeds,
        "error": None,
        "scores": {},
    }

    t0 = time.monotonic()
    try:
        sections, trace = service.search_with_trace(case["query"])
    except Exception as exc:  # noqa: BLE001
        # 链路对「一条候选都没有」的口径是显式抛异常（milvus.py），unanswerable
        # 样本要的正是这个。其余类型抛异常就是执行失败，与判负分开统计。
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_s"] = round(time.monotonic() - t0, 1)
        if case["type"] == "unanswerable":
            row["scores"]["unanswerable_raised"] = 1.0
        return row
    row["elapsed_s"] = round(time.monotonic() - t0, 1)

    if case["type"] == "unanswerable":
        # 没有种子块，三档 Recall 的分母是 0 —— 算出来会是满分（空集是任何集合的
        # 子集），混进均值就是白送的分。这类样本只判兜底行为。
        row["scores"]["unanswerable_raised"] = 0.0
        row["sections"] = [s.section for s in sections]
        return row

    # k < 种子块数的档次不计分：命中要求全中，两个种子块不可能同时排第 1，
    # `multi_hop` 的 recall@1 是结构性的 0。算进均值就等于按 multi_hop 的占比
    # 给 recall@1 加了一个固定折扣，改参数动不了它，版本间对比也读不出东西。
    row["scores"] = {
        name: scorers.recall_at_k(ids, seeds, k)
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
    row["sections"] = [s.section for s in sections]
    return row


# ── Langfuse：dataset run ─────────────────────────────────────────────────
def case_of(item) -> dict:
    """Langfuse dataset item → 跑批用的 case，`push_dataset.py` 那套切分的反向。"""
    meta = dict(item.metadata or {})
    return {
        "case_id": item.id,
        "query": item.input["query"],
        "seed_chunk_id": item.expected_output["seed_chunk_id"],
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
    """Recall 判负时，把种子块的名次带上 —— 光一个 0 在 UI 上说明不了改哪里。"""
    if not name.startswith("recall"):
        return ""
    ranks = (row.get("seed_rank") or {}).get("candidate" if name == "recall@10" else "evidence", {})
    return "种子块名次 " + "、".join(f"{cid}={rank}" for cid, rank in ranks.items())


def aggregate(*, item_results, **_):
    from langfuse import Evaluation

    rows = [r.output for r in item_results if isinstance(r.output, dict)]
    scored = [row for row in rows if row["type"] != "unanswerable" and not row["error"]]
    stats = summarize(scored)
    out = [
        Evaluation(name=name, value=stats[name], comment=f"n={stats['counted'][name]}")
        for name in RECALL
        if stats[name] is not None
    ]
    out += [Evaluation(name=name, value=stats[name]) for name in AUX if stats[name] is not None]
    return out


def run_dataset_run(args, service, wanted: set | None) -> tuple[list[dict], str | None, Progress]:
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
        row = run_case(service, case_of(item))
        progress.emit(row)
        return row

    result = client.run_experiment(
        name=args.dataset_name,
        run_name=run_name,
        description="rag-ex-1 · 三档 Recall + 辅助数",
        data=items,
        task=task,
        evaluators=[evaluate],
        run_evaluators=[aggregate],
        max_concurrency=args.concurrency,
        metadata={key: str(value) for key, value in config().items()},
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


def config() -> dict:
    """被测参数的快照。

    两次 run 的分数只有在这些值相同的前提下才可比 —— 尤其是 collection：
    重新灌库而 chunk_id 漂了，Recall 会全线暴跌，看上去像检索退化。
    """
    from llm.embedding import bge_m3
    from llm.rerank import bge_reranker
    from rag.retrieving import milvus, store
    from rag.retrieving.pipeline import assemble, recall, rerank

    return {
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


def write_result(path: Path, *, dataset: str, run_name: str, run_url: str | None,
                 elapsed_s: float, rows: list[dict]) -> None:
    scored = [row for row in rows if row["type"] != "unanswerable" and not row["error"]]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    errors = [row for row in rows if row["error"] and row["type"] != "unanswerable"]

    payload = {
        "experiment": "rag-ex-1",
        "dataset": dataset,
        "run_name": run_name,
        "run_url": run_url,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config(),
        "cases_total": len(rows),
        "cases_scored": len(scored),
        "elapsed_s": round(elapsed_s, 1),
        "summary": summarize(scored)
        | {
            "unanswerable_raised": _mean(unanswerable, "unanswerable_raised"),
            "error_rate": round(len(errors) / len(rows), 3) if rows else 0.0,
        },
        "breakdown": {key: breakdown(scored, key) for key in ("style", "type", "layer", "kind", "doc_id")},
        "cases": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_local(args, service, wanted: set | None) -> tuple[list[dict], None, Progress]:
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
        row = run_case(service, case)
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

    from llm.embedding import embedder
    from llm.rerank import reranker
    from rag.retrieving.milvus import MilvusRagService

    # 先在主线程把两个本地模型加载起来：单例是 lru_cache，多线程同时首调会各自
    # 加载一份权重，几个 G 的内存白烧。
    print("正在加载模型 …")
    embedder()
    reranker()

    service = MilvusRagService()
    wanted = set(args.cases) if args.cases else None
    runner = run_dataset_run if args.langfuse else run_local
    rows, run_url, progress = runner(args, service, wanted)

    scored = [row for row in rows if row["type"] != "unanswerable" and not row["error"]]
    summary = summarize(scored)
    print(f"\n{'=' * 70}")
    print(f"  {len(scored)} 条计分 · 用时 {progress.elapsed():.1f}s")
    for name in RECALL:
        print(f"  {name:>16}: {_fmt(summary[name])}   n={summary['counted'][name]}")
    for name in AUX:
        print(f"  {name:>16}: {summary[name]}")

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
        )
        print(f"\n  指标已写入 {out}")


if __name__ == "__main__":
    main()
