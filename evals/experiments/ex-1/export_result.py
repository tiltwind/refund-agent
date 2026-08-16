"""把 Langfuse 上某一次 dataset run 导出成 `result.json`（与跑批落盘同一份 schema）。

    python evals/experiments/ex-1/export_result.py --run v1-0a0d3c4
    python evals/experiments/ex-1/export_result.py --run v2-abc1234 --out /tmp/v2.json

`run_experiment.py` 跑完会自己写一份，这个脚本是**补拉**：结果文件丢了、换了台机器、
或者只想重新生成而不重跑（全量 27 条要七八分钟，还要花模型的钱）。

它比跑批那份多两样只有 Langfuse 才算得出的开销数据：trace 延迟与 token 用量。一条用例
对应一个 trace（多轮的每一轮是它下面的 span），所以这两个数覆盖整条用例，不是某一轮。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # 目录名带连字符，同目录只能这么 import

from evals.push_dataset import load_env  # noqa: E402
from run_experiment import RESULT_PATH, case_row, write_result  # noqa: E402


def _tokens(trace) -> int | None:
    """一条 trace 的 token 合计。usage_details 的 total 已含缓存读取部分。"""
    totals = [
        (obs.usage_details or {}).get("total", 0)
        for obs in trace.observations
        if obs.type == "GENERATION"
    ]
    return sum(totals) if totals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Langfuse 导出 dataset run 的指标结果")
    parser.add_argument("--run", required=True, help="dataset run 名，如 v1-0a0d3c4")
    parser.add_argument("--dataset", default="refund-cases-d1")
    parser.add_argument("--out", default=str(RESULT_PATH))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args()

    load_env(Path(args.env_file))

    from langfuse import get_client
    from services import telemetry

    if not telemetry.enabled():
        raise SystemExit("没配 LANGFUSE_PUBLIC_KEY / SECRET_KEY，读不到 dataset run")
    api = get_client().api

    run = api.datasets.get_run(args.dataset, args.run)
    if not run.dataset_run_items:
        raise SystemExit(f"run {args.run} 里一条用例都没有")
    item_meta = {}  # 用例的 title / priority 在 dataset item 上，run item 里没有
    page = 1
    while True:  # 接口一页最多 100 条
        batch = api.dataset_items.list(dataset_name=args.dataset, limit=100, page=page).data
        item_meta.update({item.id: (item.metadata or {}) for item in batch})
        if len(batch) < 100:
            break
        page += 1
    # 用 v2 的 scores 接口：v3 那个不返回 comment，而判分说明全在 comment 里
    summary = {
        score.name: score.value
        for score in api.scores.get_many(dataset_run_id=run.id, limit=100).data
    }
    print(f"→ {args.dataset} / {args.run}：{len(run.dataset_run_items)} 条，正在拉 trace …")

    cases = []
    project_id = None
    spans = []  # (起, 止)，用来还原墙钟耗时——用例是并发跑的，各条延迟直接相加会翻倍
    for run_item in sorted(run.dataset_run_items, key=lambda i: i.dataset_item_id):
        trace = api.trace.get(run_item.trace_id)
        # html_path 形如 /project/<project_id>/traces/<trace_id>，run 链接要用这个 id
        project_id = project_id or trace.html_path.split("/")[2]
        spans.append((trace.timestamp.timestamp(), trace.timestamp.timestamp() + (trace.latency or 0)))
        meta = item_meta.get(run_item.dataset_item_id, {})
        cases.append(
            case_row(
                case_id=run_item.dataset_item_id,
                title=meta.get("title", ""),
                priority=meta.get("priority"),
                trace_id=trace.id,
                elapsed_s=trace.latency,
                tokens=_tokens(trace),
                scores={
                    score.name: {"value": score.value, "comment": score.comment}
                    for score in api.scores.get_many(trace_id=trace.id, limit=100).data
                },
            )
        )

    base = telemetry.base_url().rstrip("/")
    out = Path(args.out)
    write_result(
        out,
        dataset=args.dataset,
        run_name=run.name,
        run_url=f"{base}/project/{project_id}/datasets/{run.dataset_id}/runs/{run.id}",
        agent={k: str(v) for k, v in (run.metadata or {}).items()},
        elapsed_s=max(end for _, end in spans) - min(start for start, _ in spans),
        summary=summary,
        cases=cases,
    )
    passed = sum(1 for case in cases if case["case_pass"])
    print(f"  {passed}/{len(cases)} 条通过 · 指标已写入 {out}")


if __name__ == "__main__":
    main()
