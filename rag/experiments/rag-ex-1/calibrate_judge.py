"""judge 校准：把上下文固定住，只让 judge 重复判定。

    python rag/experiments/rag-ex-1/calibrate_judge.py              # 3 轮，全量快照
    python rag/experiments/rag-ex-1/calibrate_judge.py --rounds 5
    python rag/experiments/rag-ex-1/calibrate_judge.py --cases R1-001 R1-002

输入是 `run_experiment.py` 落下的 `context_snapshot.json` —— 一次跑批里 judge 看到的
全部东西（问题、有序上下文、标准答案）。这里不碰 Milvus、embedding、reranker，轮与
轮之间唯一变的就是 judge 自己，分数的波动也就只能记在它头上。

出三个数：

| 数 | 说的是什么 | 怎么用 |
|---|---|---|
| 轮间均值极差 | 判定噪声底 | 两次 run 的指标差值大过它，才算链路真的动了 |
| verdict 翻转率 | 逐段/逐句判定在各轮之间判得不一致的比例 | 提示词的边界说清楚了没有 |
| 样本分数极差 | 哪几条判得最不稳 | 分歧明细写进输出，是改提示词的入口 |

温度已经是 0（`common.build_judge`），剩下的抖动来自思考模式与服务端采样。

判得稳不等于判得对。判得对要人工标注：拿同一份快照标一批 hit，与 judge 的判定算
一致率。上下文固定住之后，那份标注在改过提示词的 judge 上还能接着用。
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from rag.evals.common import judge_name, judge_reasoning, judge_structured, load_env  # noqa: E402

SNAPSHOT_PATH = HERE / "context_snapshot.json"
OUT_PATH = HERE / "calibration.json"

METRICS = ("context_precision", "context_recall")
UNIT = {"context_precision": "段", "context_recall": "句"}


# ── 判定 ──────────────────────────────────────────────────────────────────
def judge_case(case: dict) -> dict:
    """按快照里的上下文判一条，两个指标各判一次。

    `PolicySection` 只填 `section` 和 `text` —— judge 的提示词里就只出现这两项，
    分数、来源、生效日期都不进判定输入，快照也就不必存。
    """
    import judge
    from rag.retrieving.protocol import PolicySection

    sections = [PolicySection(section=c["section"], text=c["text"]) for c in case["context"]]
    return {
        "context_precision": judge.context_precision(case["question"], sections),
        "context_recall": judge.context_recall(case["ground_truth"], sections),
    }


def run_round(cases: dict, concurrency: int) -> dict:
    """一轮：每条样本判两个指标。judge 是网络调用，并发不受本地模型那把全局锁限制。"""
    ids = list(cases)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda cid: judge_case(cases[cid]), ids))
    return dict(zip(ids, results))


# ── 统计 ──────────────────────────────────────────────────────────────────
def scores(rounds: list[dict], case_id: str, metric: str) -> list[float]:
    """某条样本某个指标在各轮的分数。judge 失败的那轮没写分数，不在里面。"""
    return [
        rd[case_id][metric]["score"]
        for rd in rounds
        if rd[case_id][metric].get("score") is not None
    ]


def spread(values: list[float]) -> float | None:
    return round(max(values) - min(values), 3) if len(values) > 1 else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def flips(rounds: list[dict], case_id: str, metric: str) -> list[dict]:
    """逐条判定在各轮之间的分歧。

    输入固定，`detail` 的条数每轮相同（precision 是段数、recall 是句数），按位置
    对齐就看得出哪一段、哪一句 judge 自己都没拿准 —— 要改的提示词就在那里。
    """
    details = [rd[case_id][metric].get("detail") for rd in rounds]
    details = [d for d in details if d]
    if len(details) < 2:
        return []
    out = []
    for i in range(min(len(d) for d in details)):
        hits = [d[i]["hit"] for d in details]
        if len(set(hits)) > 1:
            out.append({
                "text": details[0][i]["text"],
                "hits": hits,
                "reasons": [d[i]["reason"] for d in details],
            })
    return out


def summarize(rounds: list[dict], case_ids: list[str]) -> dict:
    """每个指标一行：轮均值、噪声底、翻转率。

    翻转率的分母是判定位置数（所有样本的段数/句数之和），不是样本数 —— 一条样本
    判十段错一段，和判一段错一段，不是一回事。
    """
    stats = {}
    for metric in METRICS:
        round_means = [
            mean([rd[cid][metric]["score"] for cid in case_ids
                  if rd[cid][metric].get("score") is not None])
            for rd in rounds
        ]
        clean = [m for m in round_means if m is not None]
        positions = flipped = 0
        for cid in case_ids:
            details = [d for d in (rd[cid][metric].get("detail") for rd in rounds) if d]
            if len(details) > 1:
                positions += min(len(d) for d in details)
                flipped += len(flips(rounds, cid, metric))
        stats[metric] = {
            "round_means": round_means,
            "mean": mean(clean),
            "spread": spread(clean),
            "positions": positions,
            "flipped": flipped,
            "flip_rate": round(flipped / positions, 3) if positions else None,
            "judge_errors": sum(
                1 for rd in rounds for cid in case_ids if rd[cid][metric].get("error")
            ),
        }
    return stats


def unstable(rounds: list[dict], case_ids: list[str]) -> list[dict]:
    """分数在各轮之间动过的样本，极差大的排前面。"""
    out = []
    for cid in case_ids:
        for metric in METRICS:
            values = scores(rounds, cid, metric)
            gap = spread(values)
            if not gap:
                continue
            out.append({
                "case_id": cid,
                "metric": metric,
                "scores": values,
                "spread": gap,
                "flips": flips(rounds, cid, metric),
            })
    return sorted(out, key=lambda row: row["spread"], reverse=True)


# ── 入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="固定上下文，重复判定，量 judge 自己的抖动")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_PATH), help="judge 输入快照")
    parser.add_argument("--rounds", type=int, default=3, help="重复判定的轮数，默认 3")
    parser.add_argument("--cases", nargs="*", help="只判这些 case_id")
    parser.add_argument("--concurrency", type=int, default=8, help="并发用例数")
    parser.add_argument("--out", default=str(OUT_PATH), help=f"结果落盘路径，默认 {OUT_PATH.name}")
    args = parser.parse_args()

    load_env()

    path = Path(args.snapshot)
    if not path.exists():
        raise SystemExit(f"没有 {path}，先跑一次 run_experiment.py 把上下文冻下来")
    snapshot = json.loads(path.read_text(encoding="utf-8"))

    wanted = set(args.cases) if args.cases else None
    # 空上下文的条目不判：judge 对它们短路判 0，不调模型，留着只会把翻转率冲低
    cases = {
        cid: case for cid, case in snapshot["cases"].items()
        if case["context"] and (not wanted or cid in wanted)
    }
    if not cases:
        raise SystemExit(f"{path} 里没有可判的用例")
    skipped = len(snapshot["cases"]) - len(cases) if not wanted else 0

    frozen = (snapshot.get("config") or {}).get("judge_model")
    if frozen and frozen != judge_name():
        print(f"! 快照那轮的 judge 是 {frozen}，现在是 {judge_name()} —— 换 judge 等于换判定口径")

    print(f"→ 快照 {path.name} · {len(cases)} 条"
          + (f"（跳过 {skipped} 条空上下文）" if skipped else "")
          + f" · {args.rounds} 轮 · judge：{judge_name()}")

    case_ids = list(cases)
    rounds = []
    for i in range(1, args.rounds + 1):
        result = run_round(cases, args.concurrency)
        rounds.append(result)
        means = "  ".join(
            f"{short}={_fmt(mean([result[cid][metric]['score'] for cid in case_ids if result[cid][metric].get('score') is not None]))}"
            for metric, short in zip(METRICS, ("CP", "CR"))
        )
        print(f"  第 {i} 轮  {means}", flush=True)

    stats = summarize(rounds, case_ids)
    rows = unstable(rounds, case_ids)

    print(f"\n{'=' * 70}")
    for metric in METRICS:
        s = stats[metric]
        print(f"  {metric:>18}: 均值 {_fmt(s['mean'])} · 轮间极差 {_fmt(s['spread'])} · "
              f"翻转 {s['flipped']}/{s['positions']} {UNIT[metric]}"
              f"（{_fmt(s['flip_rate'])}）" + (f" · judge 失败 {s['judge_errors']} 次" if s["judge_errors"] else ""))
    print("\n  轮间极差是判定噪声底：两次 run 的指标差值大过它，才算链路真的动了")

    if rows:
        print(f"\n  分数动过的样本 {len(rows)} 条（列前 8）")
        for row in rows[:8]:
            print(f"    {row['case_id']:<10} {row['metric']:<18} "
                  f"{'/'.join(f'{v:.2f}' for v in row['scores'])} · 极差 {row['spread']:.3f}")
        first = rows[0]
        if first["flips"]:
            flip = first["flips"][0]
            print(f"\n  例：{first['case_id']} / {first['metric']} 判得不一致的一{UNIT[first['metric']]}")
            print(f"    {flip['text'][:60]}")
            for hit, reason in zip(flip["hits"], flip["reasons"]):
                print(f"      {'✓' if hit else '✗'} {reason}")
    else:
        print("\n  各轮分数完全一致")

    out = Path(args.out)
    out.write_text(json.dumps({
        "experiment": "rag-ex-1",
        "snapshot": str(path),
        "snapshot_run": snapshot.get("run_name"),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds": args.rounds,
        "n": len(cases),
        "skipped_empty_context": skipped,
        "judge": {
            "judge_model": judge_name(),
            "judge_reasoning": judge_reasoning() or "模型默认",
            "judge_structured": judge_structured() or "默认",
        },
        "retrieval_config": snapshot.get("config"),
        "metrics": stats,
        "unstable": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n  校准结果已写入 {out}")


def _fmt(value: float | None) -> str:
    return " —  " if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
