"""把跑批的 trace 导出成文件，留在仓库里。

    python rag/experiments/rag-ex-1/export_traces.py                # 按下面的口径抽 20 条
    python rag/experiments/rag-ex-1/export_traces.py --all          # 全量导出，每条都留档
    python rag/experiments/rag-ex-1/export_traces.py --cases R1-046 # 只导指定用例

Langfuse 是本地实例，链接换台机器就打不开，而报告里每条结论都要能对上现场 ——
所以现场记录留一份在仓库里（与 evals/experiments/ex-1/traces 同一个理由）。

**抽样不是随机的**：随机 20 条里大概率一条空证据都没有，而空证据恰恰是最该留档的那类。
按「报告里每个结论各留一份现场」分桶取，桶的定义见 BUCKETS，也会写进 traces/README.md。
再留三条三档全中的作对照 —— 只留坏 case 的话，读的人无从判断正常的一次检索长什么样。

每条两个文件：`.md` 人读版（六步展开，装配那步是条款全文），`.json` 机器版（trace 元信息
加全部 observation，原样保留，重新统计或做 diff 用）。

跑批带了 `--judge` 的话，`.md` 末尾多一段 judge 判定：逐条 claim 支不支撑、哪些内容单元
被判为不相关 —— 就跟在装配的正文后面，对着看的。
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from rag.evals.common import load_env  # noqa: E402

TRACES = HERE / "traces"
RESULT_PATH = HERE / "result.json"


# ── 抽样 ──────────────────────────────────────────────────────────────────
def _ranks(row: dict, where: str) -> list:
    return list((row.get("seed_rank") or {}).get(where, {}).values())


BUCKETS = (
    ("空证据", "重排后一条都不剩，Agent 拿到零证据（报告 5.1）",
     lambda r: r["scores"].get("evidence_tokens") == 0, 4),
    ("召回层丢失", "种子块根本没进 20 条候选，重排救不回来",
     lambda r: r["scores"].get("recall@10") == 0 and None in _ranks(r, "candidate"), 2),
    ("候选尾部", "种子块在候选里排 11~19，卡在 k=10 这条线上，也是 run 间抖动的来源",
     lambda r: r["scores"].get("recall@10") == 0, 2),
    ("重排后掉出前 3", "候选里有，重排后不在前 3 —— 压下去的、被阈值滤掉的、提了名次但不够的（报告四）",
     lambda r: r["scores"].get("recall@10") == 1 and r["scores"].get("recall@3") == 0, 3),
    ("重复正文", "装配把同一父块拼了两遍以上，重复率超过 50%（报告 5.3）",
     lambda r: r["scores"].get("duplicate_ratio", 0) >= 0.5, 2),
    ("unanswerable", "语料里没有的问题，链路照样返回证据（报告 5.2）",
     lambda r: r["type"] == "unanswerable", 2),
    ("multi_hop", "两个种子块都要进前 3 才算命中",
     lambda r: r["type"] == "multi_hop" and r["scores"].get("recall@3") == 1, 2),
    ("对照", "三档全中，读「正常的一次检索长什么样」用",
     lambda r: r["scores"].get("recall@1") == 1, 3),
)


REST = ("其余", "@3 命中、@1 未中的常规样本，不属于上面任何一类")


def pick(cases: list[dict], every: bool = False) -> list[tuple[str, dict]]:
    """按桶取样，先到先得。桶内按 case_id 排序，同一份 result.json 抽出来的永远是同一批。

    `every` 是 `--all`：桶的口径照旧（分组和读点都还在），只是不再截断，剩下不入桶的
    进 REST。分档表指到的用例、报告里没提到的用例，翻现场时都能找到对应的一份。
    """
    taken: set[str] = set()
    out: list[tuple[str, dict]] = []
    ordered = [r for r in sorted(cases, key=lambda r: r["case_id"]) if r.get("trace_id")]
    for name, _, match, count in BUCKETS:
        hit = [r for r in ordered if r["case_id"] not in taken and match(r)]
        for row in (hit if every else hit[:count]):
            taken.add(row["case_id"])
            out.append((name, row))
    if every:
        out += [(REST[0], r) for r in ordered if r["case_id"] not in taken]
    return out


# ── 渲染 ──────────────────────────────────────────────────────────────────
def _rank(ranks: list) -> str:
    return "/".join("×" if r is None else str(r) for r in ranks)


def headline(row: dict) -> str:
    """一句读点。人写的结论在 README 里，这里只描述这条 trace 本身发生了什么。"""
    scores = row["scores"]
    if row["type"] == "unanswerable":
        return "语料里没有的问题" + ("，链路按预期抛异常" if scores.get("unanswerable_raised") else "，链路返回了证据")
    if scores.get("evidence_tokens") == 0:
        return "重排后 0 条过阈值，返回空证据"

    cand, ev = _ranks(row, "candidate"), _ranks(row, "evidence")
    bits = []
    if scores.get("recall@10") == 0:
        bits.append("种子块没进候选" if None in cand else f"种子块在候选里排第 {_rank(cand)}")
        # @10 读候选、@3 读重排后的证据，两个序列不同，所以「@10 判负而 @3 满分」是可能的：
        # 候选第 15 被重排提进前 3。这类样本说明 k=10 那条截断线定得保守
        if scores.get("recall@3") == 1:
            bits.append("重排把它救进了前 3")
    elif scores.get("recall@3") == 0:
        if None in ev:
            bits.append(f"候选第 {_rank(cand)}，被 MIN_SCORE 滤掉")
        elif max(ev) > max(c for c in cand if c):
            bits.append(f"候选第 {_rank(cand)} → 证据第 {_rank(ev)}，重排压下去了")
        else:
            bits.append(f"候选第 {_rank(cand)} → 证据第 {_rank(ev)}，重排提了名次但仍不在前 3")
    elif scores.get("recall@1") == 1:
        bits.append("三档全中")
    else:
        bits.append(f"命中，但排第 {_rank(ev)}")

    dup = scores.get("duplicate_ratio", 0)
    if dup >= 0.4:
        bits.append(f"证据里 {dup:.0%} 是重复正文")
    return "；".join(bits)


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def render(row: dict, bucket: str, spans: dict, run_name: str) -> str:
    seeds = set(row["seed_chunk_id"])
    mark = lambda cid: " ←" if cid in seeds else ""  # noqa: E731  种子块标出来，肉眼找不动 20 行

    scores = row["scores"]
    lines = [f"# {row['case_id']} · {headline(row)}", "",
             f"抽样桶：**{bucket}**。run `{run_name}`，trace `{row['trace_id']}`。", ""]
    lines += _table(["", ""], [
        ["query", spans.get("rag.search_policy", {}).get("input", {}).get("query", "")],
        ["分类", f"{row['type']} / {row['style']} / {row['kind']} / {row['doc_id']}"],
        ["种子块", "、".join(row["seed_chunk_id"]) or "—"],
        ["判分", " · ".join(f"{k}={v}" for k, v in scores.items())],
        ["名次", json.dumps(row.get("seed_rank", {}), ensure_ascii=False)],
    ])

    step = spans.get("rag.rewrite", {}).get("output") or {}
    lines += ["", "## 1 · 改写", "",
              f"`rewritten={step.get('rewritten')}`，`needs_law={step.get('needs_law')}`", ""]
    for sub in step.get("sub_queries", []):
        lines.append(f"- [{sub['intent']}] {sub['text']}")

    step = spans.get("rag.route", {}).get("output") or {}
    lines += ["", "## 2 · 路由", ""]
    for route in step.get("routes", []):
        lines.append(f"- `{route['sub_query']}` → {route['layer_k']}，法规层权重 {route['law_weight']}")

    step = spans.get("rag.recall", {}).get("output") or {}
    cands = step.get("candidates", [])
    lines += ["", f"## 3 · 召回融合（{len(cands)} 条候选）", ""]
    lines += _table(["#", "chunk_id", "小节", "RRF", "命中来源"], [
        [i, f"`{c['chunk_id']}`{mark(c['chunk_id'])}", c["section"], c["rrf"], "、".join(c["hits"])]
        for i, c in enumerate(cands, 1)
    ])

    step = spans.get("rag.rerank", {}).get("output") or {}
    ev = step.get("evidence", [])
    lines += ["", f"## 4 · 重排（{len(cands)} → {step.get('passed')} 条过阈值，"
                  f"`MIN_SCORE={step.get('min_score')}`）", ""]
    if step.get("dropped"):
        lines += [f"被阈值砍掉 {len(step['dropped'])} 条：" +
                  "、".join(f"`{c}`" for c in step["dropped"]), ""]
    lines += _table(["#", "chunk_id", "score", "relevance", "prior", "摘录"], [
        [i, f"`{e['chunk_id']}`{mark(e['chunk_id'])}", e["score"], e["relevance"], e["prior"],
         e["excerpt"].replace("|", "\\|")]
        for i, e in enumerate(ev, 1)
    ])

    step = spans.get("rag.assemble", {}).get("output") or {}
    sections = step.get("sections", [])
    lines += ["", f"## 5 · 装配（{len(ev)} → {len(sections)} 块）", ""]
    lines += ["这一步的产物就是注入模型上下文的东西，下面是全文。", ""] if sections else [
        "**空** —— 工具层拿到的就是一个空列表，Agent 面对零证据。", ""]
    for i, sec in enumerate(sections, 1):
        lines += [f"### [E{i}] {sec['section']}", "",
                  f"`score={sec['score']:.3f}` · {sec['reason']} · {sec['source_path']}", "",
                  "```text", sec["text"], "```", ""]
    lines += judge_section(row)
    return "\n".join(lines) + "\n"


def judge_section(row: dict) -> list[str]:
    """两个 LLM 指标的逐条判定。跑批没带 `--judge` 时整段不出现。

    落在装配那步后面是有意的：上面刚看完注入模型的全文，下面就是 judge 对着这段文字
    给出的判定 —— 一个 0.6 分说明不了改哪里，要能对着正文看是哪条 claim 没撑住。
    """
    judged = row.get("judge") or {}
    if not judged:
        return []
    lines = ["", "## 6 · judge 判定", ""]

    result = judged.get("context_recall") or {}
    if result.get("skipped"):
        lines += [f"**Context Recall** —— 未判定（{result['skipped']}）", ""]
    elif result.get("error"):
        lines += [f"**Context Recall** —— judge 调用失败：`{result['error']}`，这条不计入均值", ""]
    elif result.get("detail"):
        lines += [f"**Context Recall = {result['score']}**（{result['hit']}/{result['n']} 条 claim 被支撑）", ""]
        lines += _table(["", "claim", "判定理由"], [
            ["✓" if d["supported"] else "✗", d["claim"].replace("|", "\\|"),
             d["reason"].replace("|", "\\|")]
            for d in result["detail"]
        ])
        lines.append("")

    result = judged.get("context_relevance") or {}
    if result.get("skipped"):
        lines += [f"**Context Relevance** —— 未判定（{result['skipped']}）", ""]
    elif result.get("error"):
        lines += [f"**Context Relevance** —— judge 调用失败：`{result['error']}`，这条不计入均值", ""]
    elif result.get("score") is not None:
        lines += [f"**Context Relevance = {result['score']}**（{result['hit']}/{result['n']} 个内容单元相关"
                  + ("，超过 MAX_UNITS 已截断" if result.get("truncated") else "") + f"）：{result.get('note', '')}", ""]
        if result.get("irrelevant"):
            lines += ["判为不相关的（最多列 10 条）：", ""]
            lines += [f"- {text}" for text in result["irrelevant"]]
            lines.append("")
    return lines


# ── 入口 ──────────────────────────────────────────────────────────────────
def write_index(picked: list[tuple[str, dict]], run_name: str, every: bool = False) -> None:
    lines = [
        "# traces —— 现场记录", "",
        f"从 run `{run_name}` 导出的 {len(picked)} 条，供[HTML 报告](../rag-ex-1-report.html)逐条引用。",
        "Langfuse 是本地实例，链接换台机器就打不开，所以现场留一份在仓库里。", "",
        f"重新导出：`python rag/experiments/rag-ex-1/export_traces.py{' --all' if every else ''}`。", "",
    ]
    lines += [
        "全量导出，每条用例一份。下面的分组是按报告里的结论分的，先到先得，一条只入一个组 ——",
        "空证据、召回层丢失这些是报告逐条引用的那批，`其余` 是 `@3` 命中的常规样本。", "",
    ] if every else [
        "抽样不是随机的 —— 随机 20 条里大概率一条空证据都没有，而那恰恰是最该留档的。",
        "按报告里的每个结论分桶取，最后三条是三档全中的对照：只留坏 case 的话，读的人无从",
        "判断正常的一次检索长什么样。", "",
    ]
    for name, why in [(b[0], b[1]) for b in BUCKETS] + [REST]:
        rows = [row for bucket, row in picked if bucket == name]
        if not rows:
            continue
        lines += [f"## {name}", "", why, ""]
        lines += _table(["用例", "读点", "分类"], [
            [f"[{r['case_id']}](./{r['case_id']}.md)", headline(r),
             f"{r['type']} / {r['style']} / {r['kind']}"]
            for r in rows
        ])
        lines.append("")
    lines += [
        "---", "",
        "每条两个文件：`.md` 人读版，六步展开，装配那步是条款全文；`.json` 机器版，trace 元信息",
        "加全部 observation，`input` / `output` 原样保留。", "",
        "导出的是**这一次运行的痕迹**，不是可重放的用例。用例定义在",
        "[`rag/datasets/r1/cases.jsonl`](../../../datasets/r1/cases.jsonl)。", "",
    ]
    (TRACES / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 rag-ex-1 的 trace 现场记录")
    parser.add_argument("--result", default=str(RESULT_PATH), help="读哪份结果文件")
    parser.add_argument("--cases", nargs="*", help="只导这些 case_id，跳过抽样")
    parser.add_argument("--all", action="store_true", dest="every",
                        help="全量导出，不按桶截断；分组照旧，剩下的进「其余」")
    args = parser.parse_args()

    load_env()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    cases = result["cases"]
    run_name = result["run_name"]

    if args.cases:
        wanted = set(args.cases)
        picked = [("指定", r) for r in cases if r["case_id"] in wanted and r.get("trace_id")]
    else:
        picked = pick(cases, args.every)
    if not picked:
        raise SystemExit("没有可导出的用例（结果文件里没有 trace_id？跑批要加 --langfuse）")

    from services import telemetry

    if not telemetry.enabled():
        raise SystemExit("没配 LANGFUSE_PUBLIC_KEY / SECRET_KEY，拉不到 trace")
    from langfuse import get_client

    client = get_client()
    TRACES.mkdir(exist_ok=True)

    for bucket, row in picked:
        detail = client.api.trace.get(row["trace_id"])
        obs = [o.dict() for o in detail.observations]
        spans = {o["name"]: o for o in obs}
        (TRACES / f"{row['case_id']}.md").write_text(
            render(row, bucket, spans, run_name), encoding="utf-8"
        )
        (TRACES / f"{row['case_id']}.json").write_text(
            json.dumps(
                {"case": row, "trace": {"id": detail.id, "name": detail.name,
                                        "timestamp": str(detail.timestamp)}, "observations": obs},
                ensure_ascii=False, indent=2, default=str,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"  {row['case_id']:<8} {bucket:<12} {headline(row)}")

    if not args.cases:
        write_index(picked, run_name, args.every)
    print(f"\n{len(picked)} 条 → {TRACES}")


if __name__ == "__main__":
    main()
