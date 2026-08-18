"""从 result.json 生成 HTML 报告。

    python rag/experiments/rag-ex-1/report.py                      # → rag-ex-1-report.html
    python rag/experiments/rag-ex-1/report.py --result /tmp/v2.json --out /tmp/v2.html

报告只读结果文件，不连 Langfuse —— 那是本地实例，换台机器 run 页就打不开。结论性的
文字在 [baseline-report.md] 里，这份是同一批数据的可视化：三档、分档、102 条明细。

HISTORY 那张表是人填的：result.json 只存最近一次，而 run 间的抖动幅度恰恰要几次才看得出来。
"""

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

RESULT_PATH = HERE / "result.json"
OUT_PATH = HERE / "rag-ex-1-report.html"
TRACES = HERE / "traces"

RECALL = ("recall@1", "recall@3", "recall@10")

# 同参数下跑过的 run。改了参数就别往这张表里加，那是另一条基线。
HISTORY = [
    ("离线 1", None, 0.750, 0.896, 0.212, 1260.9, 75.2),
    ("离线 2", 0.637, 0.750, 0.896, 0.208, 1251.3, 75.6),
    ("baseline-1", 0.637, 0.750, 0.875, 0.212, 1261.4, 218.2),
    ("baseline-2", 0.637, 0.760, 0.885, 0.207, 1267.0, 206.3),
]

CSS = """
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --series:#2a78d6; --track:#eceae4; --good:#0ca30c; --bad:#d03b3b;
  --warn:#fab219; --border:rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --series:#3987e5; --track:#262624; --border:rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --series:#3987e5; --track:#262624; --border:rgba(255,255,255,0.10);
}
* { box-sizing: border-box; }
body { margin:0; padding:40px 24px 80px; background:var(--page); color:var(--ink);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif; }
main { max-width:1180px; margin:0 auto; }
a { color:var(--series); text-decoration:none; }
a:hover { text-decoration:underline; }
code { font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:18px; margin:48px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--grid); }
h3 { font-size:15px; margin:0 0 10px; }
p { margin:10px 0; color:var(--ink-2); }
.sub { color:var(--muted); margin:0 0 18px; font-size:14px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
.chip { border:1px solid var(--border); border-radius:999px; padding:3px 11px;
  font-size:12.5px; color:var(--ink-2); background:var(--surface); }
.chip b { color:var(--ink); font-weight:600; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:12px; margin:20px 0 8px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
.tile .k { font-size:12px; color:var(--muted); }
.tile .v { font-size:28px; font-weight:650; letter-spacing:-0.02em; margin-top:2px; }
.tile .n { font-size:12.5px; color:var(--ink-2); }
.tile.bad .v { color:var(--bad); }
.tile.good .v { color:var(--good); }
.note { margin:18px 0; padding:14px 16px; background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--warn); border-radius:8px; font-size:14px; color:var(--ink-2); }
.note strong { color:var(--ink); }
.issue { margin:14px 0; padding:16px 18px; background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--bad); border-radius:8px; }
.issue h3 { margin-bottom:6px; }
.issue p:last-child { margin-bottom:0; }
.issue pre { background:var(--page); border:1px solid var(--grid); border-radius:6px;
  padding:10px 12px; overflow-x:auto; font:12.5px/1.6 ui-monospace,Menlo,monospace; color:var(--ink-2); }
.stack > div { margin-bottom:18px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(500px,1fr)); gap:16px 20px; }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:7px 8px; text-align:left; border-bottom:1px solid var(--grid); white-space:nowrap; }
thead th { font-size:11.5px; color:var(--muted); font-weight:600; background:var(--surface); position:sticky; top:0; }
tbody tr:hover { background:color-mix(in srgb,var(--series) 7%,transparent); }
tbody tr:last-child td { border-bottom:0; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
td.c, th.c { text-align:center; }
.id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.tag { font-size:11px; color:var(--muted); font-family:ui-monospace,Menlo,monospace; }
.mini { position:relative; height:11px; background:var(--track); border-radius:3px; min-width:46px; }
.mini i { position:absolute; left:0; top:0; height:11px; background:var(--series); border-radius:3px; }
.hitmap b { font-family:ui-monospace,Menlo,monospace; font-size:12px; padding:0 1px; }
.h1 { color:var(--good); }
.h0 { color:var(--bad); }
.hx { color:var(--muted); }
.row-bad td { background:color-mix(in srgb,var(--bad) 6%,transparent); }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:10px 2px 0; font-size:12.5px; color:var(--muted); }
footer { margin-top:56px; padding-top:16px; border-top:1px solid var(--grid); font-size:12.5px; color:var(--muted); }
"""


def esc(text) -> str:
    return html.escape(str(text))


def fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def bar_cell(value, digits: int = 3) -> str:
    """数值 + 一条到 1.0 的横条。分档表里一眼看出哪一档塌了。"""
    if value is None:
        return '<td class="c">—</td>'
    return (f'<td class="n">{fmt(value, digits)}</td>'
            f'<td><div class="mini"><i style="width:{value * 100:.1f}%"></i></div></td>')


def table(head: list[str], rows: list[str], classes: str = "") -> str:
    cells = "".join(f'<th class="{"n" if h.startswith("@") or h == "n" else ""}">{esc(h)}</th>'
                    for h in head)
    return (f'<div class="tablewrap {classes}"><table><thead><tr>{cells}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def breakdown_table(breakdown: dict, key: str) -> str:
    head = ["档位", "n", "@1", "", "@3", "", "@10", "", "token", "重复"]
    rows = []
    for name, stats in breakdown[key].items():
        cells = "".join(bar_cell(stats[metric]) for metric in RECALL)
        rows.append(
            f'<tr><td class="id">{esc(name)}</td><td class="n">{stats["n"]}</td>{cells}'
            f'<td class="n">{fmt(stats["evidence_tokens"], 0)}</td>'
            f'<td class="n">{fmt(stats["duplicate_ratio"], 3)}</td></tr>'
        )
    # 标题和表包在一个 div 里：外层是 grid，散着放会被拆成两个网格项
    return f"<div><h3>按 {esc(key)}</h3>{table(head, rows)}</div>"


def case_rows(cases: list[dict]) -> list[str]:
    marks = {1.0: '<b class="h1">1</b>', 0.0: '<b class="h0">0</b>', None: '<b class="hx">-</b>'}
    out = []
    for row in sorted(cases, key=lambda r: r["case_id"]):
        scores = row["scores"]
        hit = "".join(marks[scores.get(metric)] for metric in RECALL)
        ranks = row.get("seed_rank") or {}
        rank = "、".join(
            f'{"×" if v is None else v}' for v in (ranks.get("candidate") or {}).values()
        ) + " → " + "、".join(
            f'{"×" if v is None else v}' for v in (ranks.get("evidence") or {}).values()
        ) if ranks else "—"
        trace = (f'<a href="./traces/{row["case_id"]}.md">现场</a>'
                 if (TRACES / f'{row["case_id"]}.md').exists() else "")
        bad = " class=\"row-bad\"" if scores.get("recall@3") == 0 and row["type"] != "unanswerable" else ""
        tokens = scores.get("evidence_tokens")
        dup = scores.get("duplicate_ratio")
        out.append(
            f"<tr{bad}><td class=\"id\">{esc(row['case_id'])}</td>"
            f'<td class="tag">{esc(row["type"])} / {esc(row["style"])} / {esc(row["kind"] or "—")}</td>'
            f'<td class="id">{esc(row["doc_id"] or "—")}</td>'
            f'<td class="c hitmap">{hit}</td>'
            f'<td class="tag">{esc(rank)}</td>'
            f'<td class="n">{"—" if tokens is None else tokens}</td>'
            f'<td class="n">{"—" if dup is None else fmt(dup, 3)}</td>'
            f'<td class="c">{trace}</td></tr>'
        )
    return out


def build(result: dict) -> str:
    summary = result["summary"]
    config = result["config"]
    counted = summary["counted"]
    scored = [r for r in result["cases"] if r["type"] != "unanswerable" and not r["error"]]

    chips = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in config.items())

    tiles = "".join([
        f'<div class="tile"><div class="k">recall@1</div><div class="v">{fmt(summary["recall@1"])}</div>'
        f'<div class="n">n={counted["recall@1"]}，不含 multi_hop</div></div>',
        f'<div class="tile"><div class="k">recall@3</div><div class="v">{fmt(summary["recall@3"])}</div>'
        f'<div class="n">n={counted["recall@3"]}，实际交付水位</div></div>',
        f'<div class="tile"><div class="k">recall@10</div><div class="v">{fmt(summary["recall@10"])}</div>'
        f'<div class="n">n={counted["recall@10"]}，召回层上界</div></div>',
        f'<div class="tile"><div class="k">evidence_tokens</div>'
        f'<div class="v">{fmt(summary["evidence_tokens"], 0)}</div>'
        f'<div class="n">预算 {config["token_budget"]}，占 '
        f'{summary["evidence_tokens"] / config["token_budget"]:.0%}</div></div>',
        f'<div class="tile bad"><div class="k">duplicate_ratio</div>'
        f'<div class="v">{fmt(summary["duplicate_ratio"])}</div>'
        f'<div class="n">{sum(1 for r in scored if r["scores"].get("duplicate_ratio", 0) > 0)} / '
        f'{len(scored)} 条含重复正文</div></div>',
        f'<div class="tile bad"><div class="k">unanswerable_raised</div>'
        f'<div class="v">{fmt(summary["unanswerable_raised"])}</div>'
        f'<div class="n">6 条全部没抛异常</div></div>',
    ])

    empty = [r["case_id"] for r in scored if r["scores"].get("evidence_tokens") == 0]
    lost = [r for r in scored if r["scores"].get("recall@10") == 0]
    unranked = [r["case_id"] for r in lost if None in (r.get("seed_rank") or {}).get("candidate", {}).values()]
    tail = [f'{r["case_id"]}({max(v for v in r["seed_rank"]["candidate"].values() if v)})'
            for r in lost if r["case_id"] not in unranked]
    pressed = [r for r in scored
               if r["scores"].get("recall@10") == 1 and r["scores"].get("recall@3") == 0]
    dup_top = sorted(scored, key=lambda r: -r["scores"].get("duplicate_ratio", 0))[0]

    history_rows = [
        f'<tr><td class="id">{esc(name)}</td>'
        + "".join(f'<td class="n">{fmt(v)}</td>' for v in (r1, r3, r10, dup))
        + f'<td class="n">{tok:.1f}</td><td class="n">{sec:.1f}s</td></tr>'
        for name, r1, r3, r10, dup, tok, sec in HISTORY
    ]

    trace_count = len(list(TRACES.glob("R1-*.md"))) if TRACES.exists() else 0

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>检索评测报告 rag-ex-1 · run {esc(result["run_name"])}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>检索评测报告 rag-ex-1 · run {esc(result["run_name"])}</h1>
<p class="sub">数据集 {esc(result["dataset"])}，{result["cases_total"]} 条样本，
{result["cases_scored"]} 条计分，用时 {result["elapsed_s"]}s，写于 {esc(result["written_at"])}。
被测对象是 <code>search_policy</code> 的六步链路，不经过 Agent。</p>
<div class="chips">{chips}</div>

<h2>一、三档与辅助数</h2>
<div class="tiles">{tiles}</div>
<p><code>seed_chunk_id</code> 是下界不是真值，绝对值偏低属正常，这份数字的用途是当基准。
命中口径是<strong>全部种子块都进前 k</strong>，<code>multi_hop</code> 不给部分分；k 小于种子块数的
档次不计分，所以 <code>recall@1</code> 的分母是 {counted["recall@1"]} 而不是 {counted["recall@3"]}。</p>
<div class="note"><strong>三档读的不是同一个序列。</strong><code>recall@10</code> 读召回融合后的
20 条候选，<code>recall@3</code> / <code>recall@1</code> 读重排后的证据 —— 所以「@10 判负、@3 满分」
是可能的：候选第 15 被重排提进了前 3（R1-081、R1-048）。这类样本说明 k=10 那条线定得保守。</div>

<h2>二、分档</h2>
<div class="stack">
{breakdown_table(result["breakdown"], "style")}
{breakdown_table(result["breakdown"], "type")}
{breakdown_table(result["breakdown"], "layer")}
{breakdown_table(result["breakdown"], "kind")}
</div>
<p>口语档比书面档低一大截，而差距在 <code>@10</code> 上收窄 —— 召回层捞得到，是排序吃字面匹配，
责任在改写与稠密一路。表格块显著低于正文块，那是切片的事，不是重排的事。</p>

<h2>三、三个链路问题</h2>
<div class="issue">
<h3>重排后一条都不剩时，<code>search_policy</code> 静默返回空列表</h3>
<p>{len(empty)} 条：<code>{esc("、".join(empty))}</code>。链路对「一条候选都没有」是显式抛异常的，
但「重排后一条都不剩」走的不是这条路 —— <code>assemble([])</code> 返回空列表，Agent 面对零证据，
退回的正是凭记忆答政策那条路。</p>
<pre>rag.recall     candidates[0] = P10#004:03      ← 种子块排第 1
rag.rerank     passed=0, min_score={config["min_score"]}, dropped=[全部 20 条]
rag.assemble   sections=[]</pre>
<p>阈值偏高和空结果不报错是两件事：前者是调参，后者是口径漏洞。</p>
</div>
<div class="issue">
<h3><code>unanswerable</code> 的兜底口径不成立</h3>
<p>6 条全部没抛异常。异常只在候选为空时触发，而召回层对任何 query 都能捞回 20 条 —— 问「保价规则」
照样返回一堆退货条款。要么给链路加「最高分低于下限即判无适用条款」，要么把这类样本的判据改成
Context Relevance。</p>
</div>
<div class="issue">
<h3>重复正文占 {summary["duplicate_ratio"]:.1%}</h3>
<p>最高是 {esc(dup_top["case_id"])} 的 {dup_top["scores"]["duplicate_ratio"]:.0%}。成因在装配的分组键
<code>(parent_seq, parent_id, section_path)</code>：同一父块的子块 <code>section_path</code> 各不相同，
同一个 <code>parent_id</code> 被登记多次。<code>recall@3</code> 满分的用例里照样有 0.4 以上的重复率，
ID 级 Recall 对它完全无感。</p>
</div>

<h2>四、Recall 丢在哪一层</h2>
<div class="grid2">
<div>
<h3><code>recall@10</code> 丢的 {len(lost)} 条</h3>
{table(["情况", "条数", "用例"], [
  f'<tr><td>种子块没进候选</td><td class="n">{len(unranked)}</td>'
  f'<td class="tag">{esc("、".join(unranked))}</td></tr>',
  f'<tr><td>候选里排 11~19</td><td class="n">{len(tail)}</td>'
  f'<td class="tag">{esc("、".join(tail))}</td></tr>',
])}
<p>第二类落在 <code>CANDIDATE_LIMIT={config["candidate_limit"]}</code> 之内，重排还救得回来；
第一类救不回来，要回去看切片、块头、BM25 分析器和过滤条件。</p>
</div>
<div>
<h3><code>@10</code> 命中而 <code>@3</code> 丢的 {len(pressed)} 条</h3>
{table(["用例", "候选 → 证据"], [
  f'<tr><td class="id">{esc(r["case_id"])}</td><td class="tag">'
  + esc("、".join(str(v) for v in r["seed_rank"]["candidate"].values())) + " → "
  + esc("、".join("被滤掉" if v is None else str(v) for v in r["seed_rank"]["evidence"].values()))
  + "</td></tr>" for r in sorted(pressed, key=lambda r: r["case_id"])
])}
<p>这是 <code>RELEVANCE_WEIGHT</code> / <code>PRIOR_WEIGHT</code> / <code>DOC_PRIOR</code> 与
<code>MIN_SCORE</code> 那一组参数的账。</p>
</div>
</div>

<h2>五、run 间的抖动</h2>
<p>同一套参数跑了 {len(HISTORY)} 次。打分器是纯函数，被测链路不是 —— 改写那一步调模型，
温度 0 也不保证网关每次返回同一份拆分。定门禁容差之前先看这张表。</p>
{table(["run", "@1", "@3", "@10", "重复", "token", "耗时"], history_rows)}
<div class="note">翻转的都是种子块在候选里排 11~13 名、来回跨 <code>k=10</code> 那条线的用例。
实测幅度：<code>@3</code> ±0.010（1 条），<code>@10</code> ±0.021（2 条）。</div>

<h2>六、用例明细</h2>
<p>命中位是 <code>@1/@3/@10</code>，<span class="hitmap"><b class="h1">1</b></span> 命中、
<span class="hitmap"><b class="h0">0</b></span> 丢、<span class="hitmap"><b class="hx">-</b></span> 不适用。
名次是种子块在候选里第几 → 在证据里第几，<code>×</code> 表示不在里面。
标红的是 <code>@3</code> 判负的行。「现场」链到导出的 {trace_count} 条 trace。</p>
{table(["用例", "分类", "文档", "@1/@3/@10", "名次", "token", "重复", ""], case_rows(result["cases"]))}

<h2>七、下一步</h2>
<ol>
<li>空结果口径，与 <code>MIN_SCORE</code> 一起看。这是唯一会让 Agent 拿不到证据的问题。</li>
<li>表格块的切片，对着 <code>table</code> / <code>table+text</code> 两档跑一次对比。</li>
<li>judge 提示词与校准，接 Context Recall 与 Context Relevance。</li>
<li>改写的噪声先处理，之后才谈得上定门禁容差。</li>
</ol>

<footer>由 <code>rag/experiments/rag-ex-1/report.py</code> 从 <code>result.json</code> 生成。
结论性的文字见 <a href="./baseline-report.md">baseline-report.md</a>，现场记录见
<a href="./traces/README.md">traces/</a>。</footer>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 rag-ex-1 的 HTML 报告")
    parser.add_argument("--result", default=str(RESULT_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.write_text(build(result), encoding="utf-8")
    print(f"报告已写入 {out}（{out.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()
