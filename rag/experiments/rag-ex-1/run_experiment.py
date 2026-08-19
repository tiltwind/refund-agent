"""实验 rag-ex-1：跑数据集 r1 的检索评测，出两个 judge 指标与三个排序指标。

    python rag/experiments/rag-ex-1/run_experiment.py                     # 离线，全量 96 条
    python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001      # 只跑指定用例
    python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)

被测对象是 [`search_policy`](rag/retrieving/milvus.py)，不经过 Agent：一条 query 进去、
一组 `PolicySection` 出来。两个 judge 指标判这组上下文，三个排序指标判 `source` 在候选与
证据列表里排第几（`rank_metrics.py`，不调模型）。

两条路径，判分逻辑共用 `run_case`，落同一份 `result.json`：

| 路径 | 样本从哪来 | 上报 |
|---|---|---|
| 默认 | `rag/datasets/r1/cases.jsonl` | 无 |
| `--langfuse` | Langfuse 数据集 `retrieval-cases-r1` | 分数写回 dataset run，检索链路随 trace 上报 |

跑批还在 `result.json` 旁边落一份 `context_snapshot.json` —— 这一轮 judge 的输入，
`calibrate_judge.py` 拿它固定上下文重复判定，量 judge 自己的抖动。

前置：Milvus 起着并已灌库（`bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py`）；
`--langfuse` 还要先推数据集（`python rag/evals/push_dataset.py`）。
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

from rag.evals.common import DATASET_DIR, load_env, read_cases  # noqa: E402

RESULT_PATH = HERE / "result.json"
SNAPSHOT_PATH = HERE / "context_snapshot.json"

CONTEXTS: dict[str, dict] = {}
"""这一轮 judge 的输入，按 case_id 收集：question + 有序上下文 + ground_truth。

落一份是为了校准 judge —— 校准要量的是 judge 自己的抖动，就得把检索这个变量
拿掉，让同一份上下文反复判（`calibrate_judge.py`）。

放模块级而不是塞进 row：row 会作为 dataset run 的 output 上报，条款原文在检索
span 里已经有一份了。并发只做整键赋值，键是 case_id，各线程互不相干。
"""

JUDGE_METRICS = ("context_precision", "context_recall")
RANK_METRICS = ("candidate_hit", "hit@1", "hit@4", "mrr")
"""不调模型的排序指标（rank_metrics.py）。两组分开列：一组要判定口径相同才可比，
另一组只要数据集和 collection 没变就可比。"""

METRICS = JUDGE_METRICS + RANK_METRICS


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
            if row["failure"]:
                note = f"跑批故障 · {row['failure']}"
            else:
                note = f"{len(row['sections'])} 段 · " + " ".join(
                    f"{short}={_fmt(row['scores'].get(name))}"
                    for name, short in zip(JUDGE_METRICS, ("CP", "CR"))
                ) + f" · rank={row['retrieval']['rank_note']}"
            print(
                f"  [{self.done:>{len(str(self.total))}}/{self.total}] "
                f"{row['case_id']} {row['elapsed_s']:.1f}s · {note}",
                flush=True,
            )

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def run_case(service, case: dict) -> dict:
    """跑一条用例并判分。

    检索有三种出口，记账各不相同：

    | 出口 | 判分 | 计入均值 |
    |---|---|---|
    | 正常返回 | 判 | 是 |
    | `RetrievalError`（没有可交付证据） | 上下文为空，两个 judge 指标判 0 | 是 |
    | 其他异常（跑批故障） | 不判 | 否，进 `failures` 清单 |

    第三行是新分出来的。之前它与第二行共用一个 `except Exception`，
    一次 tokenizer 竞态（`RuntimeError: Already borrowed`）被记成了「检索没召回」——
    分母里多出几条判 0 的样本，均值掉下去，看上去像检索退化。这与 judge 失败
    不写分数是同一条口径：故障不记到被测对象头上。
    """
    import judge
    import rank_metrics
    from rag.retrieving.protocol import RetrievalError, RetrievalTrace

    row = {
        "case_id": case["case_id"],
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "doc_id": doc_of(case),
        "error": None,
        "failure": None,
        "sections": [],
        "scores": {},
    }

    t0 = time.monotonic()
    trace = RetrievalTrace()
    try:
        sections, trace = service.search_with_trace(case["question"])
    except RetrievalError as exc:
        # 链路对「重排后一条证据都不剩」的口径是显式抛异常（milvus.py）。这不是
        # 跑批故障，是检索结果：上下文为空，两个 judge 指标都判 0，留在均值里。
        row["error"] = f"{type(exc).__name__}: {exc}"
        trace = exc.trace or trace  # 空结果的候选序列在异常里带着，排序指标照样算
        sections = []
    except Exception as exc:  # noqa: BLE001
        row["failure"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_s"] = round(time.monotonic() - t0, 1)
        row["retrieval"] = {"candidate_ids": [], "evidence_ids": [], "rank_note": "跑批故障"}
        return row  # 判分整条跳过：这条样本这一轮没有结果，不是结果不好
    row["elapsed_s"] = round(time.monotonic() - t0, 1)
    row["sections"] = [s.section for s in sections]
    CONTEXTS[case["case_id"]] = {
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        # 有序 —— Context Precision 的位置权重落在这个次序上，快照乱序等于换了输入
        "context": [{"section": s.section, "text": s.text} for s in sections],
    }

    source = case.get("source") or []
    row["retrieval"] = {
        # 两条 ID 序列落盘：排序指标读它，事后复核「这条到底卡在召回还是阈值」也读它
        "candidate_ids": trace.candidate_ids,
        "evidence_ids": trace.evidence_ids,
        "rank_note": rank_metrics.explain(source, trace.candidate_ids, trace.evidence_ids),
    }
    row["scores"].update(rank_metrics.rank_metrics(source, trace.candidate_ids, trace.evidence_ids))

    _record(row, "context_precision", judge.context_precision(case["question"], sections))
    _record(row, "context_recall", judge.context_recall(case["ground_truth"], sections))
    return row


def _record(row: dict, name: str, result: dict) -> None:
    """判定结果落两处：分数进 `scores` 参与聚合，逐条理由进 `judge` 供翻查。

    judge 调用失败不写分数 —— 写个 0 会被均值当成「检索没召回」，把模型故障记到
    检索头上。失败的那条留在 `judge` 里，跑批末尾按数量汇报。
    """
    row.setdefault("judge", {})[name] = result
    if result["score"] is not None:
        row["scores"][name] = result["score"]


def doc_of(case: dict) -> str:
    """样本出自哪篇文档。只用于分档，不参与判分 —— 判分不看 ID（4 · 三）。"""
    docs = sorted({sid.split("#")[0] for sid in case.get("source") or []})
    return "+".join(docs) or "-"


# ── Langfuse：dataset run ─────────────────────────────────────────────────
def case_of(item) -> dict:
    """Langfuse dataset item → 跑批用的 case，`push_dataset.py` 那套切分的反向。"""
    return {
        "case_id": item.id,
        "question": item.input["question"],
        "ground_truth": item.expected_output["ground_truth"],
        "source": (item.metadata or {}).get("source") or [],
    }


def evaluate(*, output, **_):
    """把 `run_case` 已经算好的分搬成 Evaluation。

    判分不在这里做 —— 两条路径必须给出同一个数字，打分逻辑只能有一份，
    就在 `run_case` 里。这个函数只负责往 Langfuse 上搬。
    """
    from langfuse import Evaluation

    out = [
        Evaluation(name=name, value=value, comment=_why(output, name))
        for name, value in output["scores"].items()
    ]
    out.append(
        Evaluation(name="run_error", value=1.0 if output["error"] else 0.0,
                   comment=output["error"] or "ok")
    )
    if output["failure"]:
        # 跑批故障单列一个名字：它与 run_error 混在一起，UI 上就分不出
        # 「检索交不出证据」和「这条根本没跑成」
        out.append(Evaluation(name="run_failure", value=1.0, comment=output["failure"]))
    return out


def _why(row: dict, name: str) -> str:
    """判负时把判定理由带上 —— 光一个 0 在 UI 上说明不了改哪里。"""
    if name in RANK_METRICS:
        return (row.get("retrieval") or {}).get("rank_note", "")
    result = (row.get("judge") or {}).get(name, {})
    missed = [d["text"][:40] for d in result.get("detail", []) if not d["hit"]]
    unit = "段条款相关" if name == "context_precision" else "句被支撑"
    return f"{result.get('hit')}/{result.get('n')} {unit}" + (
        "；没中：" + "｜".join(missed[:3]) if missed else ""
    )


def aggregate(*, item_results, **_):
    from langfuse import Evaluation

    rows = [r.output for r in item_results if isinstance(r.output, dict)]
    stats = summarize(rows)
    return [
        Evaluation(name=name, value=stats[name], comment=f"n={stats['counted'][name]}")
        for name in METRICS
        if stats[name] is not None
    ]


def run_dataset_run(args, service, wanted: set | None) -> tuple[list[dict], str | None, Progress]:
    """走 Langfuse 的 run_experiment：样本从 dataset 拉，分数写回 dataset run。

    样本不从本地 jsonl 读 —— dataset run 要挂在 Langfuse 那份数据集上，版本对比才有
    可比的对象。两份不一致时以推上去的那份为准。
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
        description="rag-ex-1 · Context Precision + Context Recall",
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
        # trace_id 落盘：报告里指到某一条时，能直接跳到那条 trace 看检索链路
        row["trace_id"] = item_result.trace_id
        rows.append(row)
    return rows, result.dataset_run_url, progress


def run_local(args, service, wanted: set | None) -> tuple[list[dict], None, Progress]:
    """离线跑批：样本从 jsonl 读，什么都不上报。"""
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


# ── 聚合 ──────────────────────────────────────────────────────────────────
def _fmt(value: float | None) -> str:
    return " —  " if value is None else f"{value:.3f}"


def _mean(rows: list[dict], name: str) -> float | None:
    values = [row["scores"][name] for row in rows if name in row["scores"]]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    # 分母记下来才分得清一档的涨跌是链路变了还是样本少了 —— judge 失败的那条
    # 不写分数，它就不在分母里
    return {
        "n": len(rows),
        **{name: _mean(rows, name) for name in METRICS},
        "counted": {name: sum(1 for row in rows if name in row["scores"]) for name in METRICS},
    }


def breakdown(rows: list[dict]) -> dict:
    """按样本出处的文档切均值。总均值只用来看趋势，能指向改哪一篇的是这张表。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["doc_id"] or "-"].append(row)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def config() -> dict:
    """被测参数的快照。

    两次 run 的分数只有在这些值相同的前提下才可比 —— 尤其是 judge 模型与
    collection：换 judge 等于换判定口径，重新灌库等于换被检索的语料。
    """
    from llm.embedding import bge_m3
    from llm.rerank import bge_reranker
    from rag.retrieving import milvus, store
    from rag.retrieving.pipeline import assemble, recall, rerank

    from rag.evals.common import judge_name, judge_reasoning, judge_structured

    return {
        "judge_model": judge_name(),
        "judge_reasoning": judge_reasoning() or "模型默认",
        "judge_structured": judge_structured() or "默认",
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


def empty_context(rows: list[dict]) -> list[str]:
    """检索跑成了、但一条证据都没交付的用例。"""
    return [row["case_id"] for row in rows if not row["failure"] and not row["sections"]]


def failures(rows: list[dict]) -> list[str]:
    """跑批故障的清单。这些条目一个分数都没写，不在任何一个均值的分母里。"""
    return [f"{row['case_id']}: {row['failure']}" for row in rows if row["failure"]]


def judge_errors(rows: list[dict]) -> list[str]:
    """judge 调用失败的清单。这些条目没写分数，均值的分母也就少一条 —— 清单报出来，
    才看得出这一轮的指标算在多少条样本上。"""
    return [
        f"{row['case_id']}/{name}: {result['error']}"
        for row in rows
        for name, result in (row.get("judge") or {}).items()
        if result.get("error")
    ]


def write_result(path: Path, *, dataset: str, run_name: str, run_url: str | None,
                 elapsed_s: float, rows: list[dict]) -> None:
    payload = {
        "experiment": "rag-ex-1",
        "dataset": dataset,
        "run_name": run_name,
        "run_url": run_url,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config(),
        "elapsed_s": round(elapsed_s, 1),
        "summary": summarize(rows) | {
            # 空上下文的条数：这批的两个 judge 指标都是 0，且它们是同一个原因造成的，
            # 混在均值里看不出来。跑批故障不算在内 —— 那条压根没检索
            "empty_context": empty_context(rows),
            "failures": failures(rows),
            "judge_errors": judge_errors(rows),
        },
        "breakdown": breakdown(rows),
        "cases": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── 落盘 ──────────────────────────────────────────────────────────────────
def write_snapshot(path: Path, *, dataset: str, run_name: str) -> None:
    """把这一轮 judge 的输入冻下来：question + 有序上下文 + ground_truth。

    这三样就是 judge 看到的全部 —— ground_truth 的拆句（`judge.sentences`）是纯
    函数，判定单元本来就可复现，缺的只是条款原文，`result.json` 里的 `sections`
    只有标题。

    `config` 一起写：快照是这套参数下的检索产物，换 collection、换 embedding 或
    重排参数动了，就得重新冻一份。跑批故障的那条不在里面，它这一轮没有上下文。
    """
    payload = {
        "experiment": "rag-ex-1",
        "dataset": dataset,
        "run_name": run_name,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config(),
        "cases": CONTEXTS,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── 入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="跑 r1 检索评测，算两个 judge 指标与三个排序指标")
    parser.add_argument("dataset", nargs="?", default=str(DATASET_DIR), help="数据集目录（离线路径用）")
    parser.add_argument("--langfuse", action="store_true",
                        help="样本从 Langfuse 数据集拉，分数写回 dataset run，检索链路上报 trace")
    parser.add_argument("--dataset-name", default="retrieval-cases-r1",
                        help="Langfuse 上的数据集名，配合 --langfuse")
    parser.add_argument("--run-name", help="默认 <数据集>-<条数>cases；版本对比时传 git sha")
    parser.add_argument("--cases", nargs="*", help="只跑这些 case_id")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="并发用例数；本地模型的前向与 tokenizer 都被一把全局锁串起来，"
                             "调高它并行的是改写与 judge 那些网络调用")
    parser.add_argument("--out", help=f"指标落盘路径，默认 {RESULT_PATH.name}；"
                                      "只跑子集时默认不写，免得覆盖全量结果")
    args = parser.parse_args()

    load_env()

    # 离线跑批不上报检索 trace：那种 trace 不挂在任何 dataset run 上，堆在项目里只是
    # 噪音。必须在 import 检索链路之前设 —— SPAN 是模块常量，import 时就求值了。
    if not args.langfuse:
        os.environ["REFUND_AGENT_RAG_SPAN"] = "off"

    from rag.evals.common import judge_name

    print(f"→ judge：{judge_name()}")

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

    summary = summarize(rows)
    print(f"\n{'=' * 70}")
    print(f"  {len(rows)} 条 · 用时 {progress.elapsed():.1f}s")
    for name in JUDGE_METRICS:
        print(f"  {name:>18}: {_fmt(summary[name])}   n={summary['counted'][name]}")
    print(f"  {'—' * 20}")
    for name in RANK_METRICS:
        print(f"  {name:>18}: {_fmt(summary[name])}   n={summary['counted'][name]}")
    if empty := empty_context(rows):
        more = " …" if len(empty) > 8 else ""
        print(f"\n  空上下文 {len(empty)} 条（两个 judge 指标都判 0）：{'、'.join(empty[:8])}{more}")
    if broken := failures(rows):
        print(f"\n  跑批故障 {len(broken)} 条（一个分数都没写，不在任何均值的分母里）")
        for line in broken[:5]:
            print(f"    {line}")
    if errors := judge_errors(rows):
        print(f"\n  judge 失败 {len(errors)} 次（这些条目不计入均值）")
        for line in errors[:5]:
            print(f"    {line}")

    print("\n  按文档分档（只列 hit@4 最低的 8 篇）")
    ranked = sorted(breakdown(rows).items(), key=lambda kv: (kv[1]["hit@4"] or 0, kv[1]["mrr"] or 0))
    for value, stats in ranked[:8]:
        cells = "  ".join(
            f"{short}={_fmt(stats[name])}"
            for name, short in zip(("context_precision", "context_recall", "hit@4", "mrr"),
                                   ("CP", "CR", "hit@4", "mrr"))
        )
        print(f"    {value:<12} n={stats['n']:<4} {cells}")

    if run_url:
        print(f"\n  {run_url}")

    out = Path(args.out) if args.out else (None if args.cases else RESULT_PATH)
    if out:
        dataset = args.dataset_name if args.langfuse else str(Path(args.dataset).resolve().relative_to(ROOT))
        run_name = args.run_name or f"{Path(dataset).name}-{len(rows)}cases"
        write_result(
            out,
            dataset=dataset,
            run_name=run_name,
            run_url=run_url,
            elapsed_s=progress.elapsed(),
            rows=rows,
        )
        print(f"\n  指标已写入 {out}")

        # 快照与 result 同去留：跑子集时不写，免得半份快照盖掉全量那份
        snapshot = out.with_name(SNAPSHOT_PATH.name)
        write_snapshot(snapshot, dataset=dataset, run_name=run_name)
        print(f"  judge 输入快照已写入 {snapshot}（{len(CONTEXTS)} 条，校准 judge 用）")


if __name__ == "__main__":
    main()
