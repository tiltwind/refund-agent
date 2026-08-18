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
OBS = ("context_recall", "context_relevance")

# 同参数下跑过的 run。改了参数就别往这张表里加，那是另一条基线。
HISTORY = [
    ("离线 1", None, 0.750, 0.896, 0.212, 1260.9, 75.2),
    ("离线 2", 0.637, 0.750, 0.896, 0.208, 1251.3, 75.6),
    ("baseline-1", 0.637, 0.750, 0.875, 0.212, 1261.4, 218.2),
    ("baseline-2", 0.637, 0.760, 0.885, 0.207, 1267.0, 206.3),
    ("baseline-4", 0.625, 0.760, 0.885, 0.206, 1229.1, 2940.1),
    ("baseline-5", 0.650, 0.760, 0.896, 0.208, 1255.2, 600.2),
]
"""编号跳过 3：那次跑到 25 条中止（换 judge 端点），没有 run 级的数。`baseline-4` 与
`baseline-5` 的耗时是带 `--judge` 的 —— 两次 judge 调用占了绝大部分，与前几行不可比。
两者差近五倍是 judge 换了模型并关掉思考（deepseek-v4-flash → v4-pro + REASONING=none），
被测链路没动，这张表看的是三档。"""

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
.note pre, .issue pre { background:var(--page); border:1px solid var(--grid); border-radius:6px;
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
td.wrap { white-space:normal; min-width:240px; max-width:460px; }
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
        return '<td class="n">—</td><td></td>'
    return (f'<td class="n">{fmt(value, digits)}</td>'
            f'<td><div class="mini"><i style="width:{value * 100:.1f}%"></i></div></td>')


def table(head: list[str], rows: list[str], classes: str = "") -> str:
    cells = "".join(f'<th class="{"n" if h.startswith("@") or h == "n" else ""}">{esc(h)}</th>'
                    for h in head)
    return (f'<div class="tablewrap {classes}"><table><thead><tr>{cells}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def breakdown_table(breakdown: dict, key: str, judged: bool = False) -> str:
    """一档一行。带 `--judge` 跑的话多两列 —— 三个指标同一行看才读得出归因（见第三节）。"""
    head = ["档位", "n", "@1", "", "@3", "", "@10", "", "token", "重复"]
    head += ["CR", "CRel"] if judged else []
    rows = []
    for name, stats in breakdown[key].items():
        cells = "".join(bar_cell(stats[metric]) for metric in RECALL)
        obs = "".join(f'<td class="n">{fmt(stats.get(metric))}</td>' for metric in OBS) if judged else ""
        rows.append(
            f'<tr><td class="id">{esc(name)}</td><td class="n">{stats["n"]}</td>{cells}'
            f'<td class="n">{fmt(stats["evidence_tokens"], 0)}</td>'
            f'<td class="n">{fmt(stats["duplicate_ratio"], 3)}</td>{obs}</tr>'
        )
    # 标题和表包在一个 div 里：外层是 grid，散着放会被拆成两个网格项
    return f"<div><h3>按 {esc(key)}</h3>{table(head, rows)}</div>"


def case_rows(cases: list[dict], judged: bool = False) -> list[str]:
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
        # judge 跳过的（空证据、没有 claim）与 judge 失败的都没有分数，这里一律是 —，
        # 两者的区别在 result.json 的 judge 字段里，也在导出的 trace 末尾
        obs = "".join(f'<td class="n">{fmt(scores.get(m))}</td>' for m in OBS) if judged else ""
        out.append(
            f"<tr{bad}><td class=\"id\">{esc(row['case_id'])}</td>"
            f'<td class="tag">{esc(row["type"])} / {esc(row["style"])} / {esc(row["kind"] or "—")}</td>'
            f'<td class="id">{esc(row["doc_id"] or "—")}</td>'
            f'<td class="c hitmap">{hit}</td>'
            f'<td class="tag">{esc(rank)}</td>'
            f'<td class="n">{"—" if tokens is None else tokens}</td>'
            f'<td class="n">{"—" if dup is None else fmt(dup, 3)}</td>{obs}'
            f'<td class="c">{trace}</td></tr>'
        )
    return out


def judge_tiles(summary: dict, scored: list[dict]) -> list[str]:
    """两个 LLM 指标的卡片。跑批没带 `--judge` 时结果文件里没有这两个键，整块略过。

    它们与三档 Recall 分开标注：judge 有噪声，不参与 pass / fail，读的是趋势。
    """
    labels = {
        "context_recall": "参考答案的 claim 被支撑的比例",
        "context_relevance": "检回内容里相关句子的占比，看的是相对变化",
    }
    tiles = []
    for name, note in labels.items():
        if summary.get(name) is None:
            continue
        n = sum(1 for r in scored if name in r["scores"])
        tiles.append(
            f'<div class="tile"><div class="k">{name}</div><div class="v">{fmt(summary[name])}</div>'
            f'<div class="n">n={n}，{note}</div></div>'
        )
    return tiles


def quadrants(scored: list[dict]) -> str:
    """Recall@3 × Context Recall 的四象限。

    这张表是这两个指标存在的理由：只报 Recall 的话，「检索失败」和「召回了等价条款、
    种子 ID 判负」长得一模一样，调参会朝着拟合标注走。CR 的分界取 0.8 —— 一条参考
    答案拆四五条 claim，撑住四条以上算撑得住。
    """
    rows = []
    cells = [
        ("@3 判负 · CR 低", lambda r: r["r3"] == 0 and r["cr"] < 0.8,
         "真的没召回", "召回层，先看 <code>recall@10</code> 分档"),
        ("@3 判负 · CR 高", lambda r: r["r3"] == 0 and r["cr"] >= 0.8,
         "召回了等价条款，种子 ID 判负", "检索没问题，是 <code>seed_chunk_id</code> 不全"),
        ("@3 命中 · CR 低", lambda r: r["r3"] == 1 and r["cr"] < 0.8,
         "种子块召回了但答案撑不住", "切片切碎了，或装配把关键段截断了"),
        ("@3 命中 · CR 高", lambda r: r["r3"] == 1 and r["cr"] >= 0.8,
         "检索没问题", "答复仍然错的话，问题在生成阶段"),
    ]
    both = [{"id": r["case_id"], "r3": r["scores"]["recall@3"], "cr": r["scores"]["context_recall"],
             "crel": r["scores"].get("context_relevance")}
            for r in scored
            if "recall@3" in r["scores"] and "context_recall" in r["scores"]]
    for name, match, verdict, where in cells:
        hit = [r for r in both if match(r)]
        share = f"{len(hit) / len(both):.0%}" if both else "—"
        rows.append(
            f'<tr><td>{esc(name)}</td><td class="n">{len(hit)}</td><td class="n">{share}</td>'
            f'<td>{verdict}</td><td>{where}</td>'
            f'<td class="tag">{esc("、".join(r["id"] for r in hit[:6]))}</td></tr>'
        )
    return table(["象限", "n", "占比", "判断", "动哪里", "用例"], rows)


def unsupported_rows(scored: list[dict], limit: int = 12) -> list[str]:
    """撑不住的 claim 最多的那些用例。一个 0.6 分说明不了改哪里，要能翻到是哪条。"""
    ranked = sorted(
        (r for r in scored if (r.get("judge") or {}).get("context_recall", {}).get("detail")),
        key=lambda r: (r["scores"].get("context_recall", 1), -r["judge"]["context_recall"]["n"]),
    )
    out = []
    for row in ranked[:limit]:
        result = row["judge"]["context_recall"]
        missed = [d for d in result["detail"] if not d["supported"]]
        if not missed:
            continue
        marks = {1.0: '<b class="h1">1</b>', 0.0: '<b class="h0">0</b>', None: '<b class="hx">-</b>'}
        out.append(
            f'<tr><td class="id">{esc(row["case_id"])}</td>'
            f'<td class="c hitmap">{marks[row["scores"].get("recall@3")]}</td>'
            f'<td class="n">{fmt(row["scores"].get("context_recall"))}</td>'
            f'<td class="n">{result["hit"]}/{result["n"]}</td>'
            f'<td class="wrap">{esc(missed[0]["claim"])}</td>'
            f'<td class="tag wrap">{esc(missed[0]["reason"])}</td></tr>'
        )
    return out


def judge_notes(result: dict, scored: list[dict]) -> str:
    """judge 侧的健康度：失败几次、跳过几条、unanswerable 的相关度对照。"""
    summary = result["summary"]
    errors = summary.get("judge_errors") or []
    skipped = sum(
        1 for row in result["cases"]
        for verdict in (row.get("judge") or {}).values() if verdict.get("skipped")
    )
    normal = summary.get("context_relevance")
    lonely = summary.get("unanswerable_relevance")
    una = [row for row in result["cases"] if row["type"] == "unanswerable"]
    empty = [row for row in una if not row.get("sections")]
    judged = [row for row in una if "context_relevance" in row["scores"]]
    # 分母要写出来：5 条是空证据、没东西可判，剩下的才进这个均值。不写 n 的话
    # 「相关度 0.000」读起来像 6 条的结论，实际是一条样本
    gap = (f'语料里没有的问题，{len(una)} 条里 <b>{len(empty)}</b> 条被重排全滤成空证据，'
           f'剩下 {len(judged)} 条的相关度是 <b>{fmt(lonely)}</b>，正常样本是 <b>{fmt(normal)}</b>——'
           + ("低于正常样本，这个数可以当兜底判据用。"
              if lonely is not None and normal is not None and lonely < normal
              else "并没有明显低于正常样本，拿它当兜底判据还不成立。")
           + f'{len(una)} 条一条都没抛异常：空证据与「无适用条款」在链路上是同一个返回值。'
           ) if una else ""
    return (
        f'<div class="note"><strong>judge 侧的健康度。</strong>'
        f'调用失败 {len(errors)} 次（失败的不写分数，从均值里少掉，不是记 0 —— '
        f'写个 0 会被均值当成「检索没召回」）；跳过 {skipped} 次（空证据、没有 claim，'
        f'那是分母为 0，不是故障）。{gap}'
        + (f'<pre>{esc(chr(10).join(errors[:8]))}</pre>' if errors else "")
        + '</div>'
    )


def llm_metrics(result: dict, scored: list[dict]) -> str:
    """第三节的正文。三档 Recall 回答「该召回的召回了吗」，这两个回答另外两件事：
    检回的东西撑不撑得住参考答案、里面有多少是废的。校准之前只作观察，不进门禁。"""
    summary = result["summary"]
    unsupported = unsupported_rows(scored)
    return f"""
<p><code>Context Recall</code> = 被检回上下文支撑的 claim 数 / 参考答案拆出的 claim 总数，
claim 跟着样本走（<code>cases.jsonl</code> 的 <code>claims</code> 字段），不在跑批时现拆 ——
分母得在两次 run 之间保持一致，否则涨跌读不出是链路变了还是拆分变了。
<code>Context Relevance</code> = 与 query 相关的内容单元数 / 检回的内容单元总数，内容单元取句子，
表格行整行算一个。两个都判在 <code>PolicySection.text</code> 上，那是真正注入模型上下文的东西。</p>
<div class="note"><strong>这两个数现在只记录，不参与 pass / fail。</strong>judge 本身还没校准
（与人工标注的一致率、判两遍的自一致性都还没测），而链路自身也有非确定性 —— 两者从跑批结果上
分不开。<code>Context Relevance</code> 的绝对值本来就不该追求高：一条 <code>PolicySection</code>
是回填后的完整小节，里面混着不相关的句子是父块回填故意带进来的。它的用途是横向对比：
<code>top_k</code> 调大时 <code>Context Recall</code> 上升、它下降，两个一起看才知道是净赚还是净亏。</div>

<h3><code>recall@3</code> × <code>Context Recall</code> 的四象限</h3>
{quadrants(scored)}
<p>第二行是这两个指标存在的理由：种子 ID 判负、但检回的上下文照样撑得住参考答案 ——
链路召回的是等价条款。只报 Recall 的话它和「真的没召回」长得一模一样，会导致朝着拟合标注去调参。
第三行相反：种子块进了前 3，答案却撑不住，那是切片或装配的账。</p>

<h3>撑不住的 claim</h3>
{table(["用例", "@3", "CR", "撑住/总数", "第一条没撑住的 claim", "judge 的理由"], unsupported)
 if unsupported else "<p>没有判负的 claim。</p>"}
<p>列的是 <code>Context Recall</code> 最低的那些，每条只显示第一条没撑住的 claim。
逐条判定在 <code>result.json</code> 的 <code>judge</code> 字段里，也在导出的 trace 末尾 ——
对着装配那步的正文看，才知道是没召回还是召回了但被截断。</p>

{judge_notes(result, scored)}
"""


def build(result: dict) -> str:
    summary = result["summary"]
    config = result["config"]
    counted = summary["counted"]
    scored = [r for r in result["cases"] if r["type"] != "unanswerable" and not r["error"]]
    # 跑批没带 --judge 时结果文件里没有这两个键，相关的列与整节都不出现
    judged = summary.get("context_recall") is not None

    chips = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in config.items())
    judge_line = (f'两个 LLM 指标由 <code>{esc(config["judge_model"])}</code> 判定。'
                  if judged else "本次跑批没带 <code>--judge</code>，两个 LLM 指标未计算。")
    llm_section = llm_metrics(result, scored) if judged else (
        '<p>本次跑批没带 <code>--judge</code>。补算：'
        '<code>python rag/experiments/rag-ex-1/run_experiment.py --judge</code>。</p>')

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
    ] + judge_tiles(summary, scored))

    una = [r for r in result["cases"] if r["type"] == "unanswerable"]
    una_empty = sum(1 for r in una if not r.get("sections"))
    una_judged = len(una) - una_empty
    una_rel = fmt(summary.get("unanswerable_relevance"))
    empty_rows = [r for r in scored if r["scores"].get("evidence_tokens") == 0]
    empty_cut = sorted((r for r in empty_rows if r["scores"].get("recall@10") == 1),
                       key=lambda r: r["case_id"])
    empty_miss = sorted((r for r in empty_rows if r["scores"].get("recall@10") != 1),
                        key=lambda r: r["case_id"])
    cut_list = "、".join(
        f'{r["case_id"]}（候选第 {min(v for v in r["seed_rank"]["candidate"].values() if v)}）'
        for r in empty_cut)
    miss_list = "、".join(r["case_id"] for r in empty_miss)
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
被测对象是 <code>search_policy</code> 的六步链路，不经过 Agent。{judge_line}</p>
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
{breakdown_table(result["breakdown"], "style", judged)}
{breakdown_table(result["breakdown"], "type", judged)}
{breakdown_table(result["breakdown"], "layer", judged)}
{breakdown_table(result["breakdown"], "kind", judged)}
</div>
<p>口语档比书面档低一大截，而差距在 <code>@10</code> 上收窄 —— 召回层捞得到，是排序吃字面匹配，
责任在改写与稠密一路。表格块显著低于正文块，那是切片的事，不是重排的事。</p>

<h2>三、两个 LLM 指标</h2>
{llm_section}

<h2>四、四个链路问题</h2>
<div class="issue">
<h3><code>MIN_SCORE = {config["min_score"]}</code> 把召回层排第一的种子块也滤掉了</h3>
<p>{len(empty_cut)} 条重排后一条证据都不剩，而它们的 <code>recall@10</code> 全部判正：
<code>{esc(cut_list)}</code>。种子块在召回层排在最前面，重排给出的分低于
{config["min_score"]}，20 条候选连同它一起被砍光，<code>recall@3</code> 与 <code>recall@1</code>
各因此丢 {len(empty_cut)} 条。</p>
<pre>rag.recall     candidates[0] = P10#004:03      ← 种子块排第 1
rag.rerank     passed=0, min_score={config["min_score"]}, dropped=[全部 20 条]
rag.assemble   sections=[]</pre>
<p>这是调参：阈值往下调能救回这几条，代价是低分噪声一并进证据。它与第五节那张「候选 → 证据」
表动的是同一组参数，要一起调、一起看。</p>
</div>
<div class="issue">
<h3>重排后一条不剩时，<code>search_policy</code> 返回空列表而不抛异常</h3>
<p>候选为空时链路是显式抛异常的，重排后为空时不是 —— <code>assemble([])</code> 返回空列表，
工具层拿到零证据，Agent 退回凭记忆答政策那条路。这次有 {len(empty_rows)} 条走到这里：
上一条那 {len(empty_cut)} 条，加上种子块本就没进候选的 <code>{esc(miss_list)}</code>。</p>
<p>这不是上一条的另一种说法。把 <code>MIN_SCORE</code> 调低只是让这几条不再触发，口径不变：
<strong>「一条可用证据都没有」与正常结果在返回值上仍是同一种东西</strong>，调用方无从区分。
下面 <code>unanswerable</code> 那 6 条撞的也是这一条。</p>
</div>
<div class="issue">
<h3><code>unanswerable</code> 的兜底口径不成立</h3>
<p>6 条全部没抛异常（<code>unanswerable_raised = 0</code>）。异常只在候选为空时触发，而召回层对任何
query 都能捞回 20 条。实际发生的是另一回事：{una_empty} 条被 <code>MIN_SCORE</code> 全滤成空证据，
剩下 {una_judged} 条返回了不相关的条款、相关度判出来是 {una_rel}。</p>
<p>也就是说链路对这类问题多半已经「什么都没给」，但给不出与前一个问题的区别 ——
<strong>空证据与「语料里没有适用条款」在返回值上是同一个空列表</strong>。要么给链路加一个
「最高分低于下限即判无适用条款」的显式结论，要么把这类样本的判据改成 Context Relevance。</p>
</div>
<div class="issue">
<h3>重复正文占 {summary["duplicate_ratio"]:.1%}</h3>
<p>最高是 {esc(dup_top["case_id"])} 的 {dup_top["scores"]["duplicate_ratio"]:.0%}。成因在装配的分组键
<code>(parent_seq, parent_id, section_path)</code>：同一父块的子块 <code>section_path</code> 各不相同，
同一个 <code>parent_id</code> 被登记多次。<code>recall@3</code> 满分的用例里照样有 0.4 以上的重复率，
ID 级 Recall 对它完全无感。</p>
</div>

<h2>五、Recall 丢在哪一层</h2>
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

<h2>六、run 间的抖动</h2>
<p>同一套参数跑了 {len(HISTORY)} 次。打分器是纯函数，被测链路不是 —— 改写那一步调模型，
温度 0 也不保证网关每次返回同一份拆分。定门禁容差之前先看这张表。</p>
{table(["run", "@1", "@3", "@10", "重复", "token", "耗时"], history_rows)}
<div class="note">翻转的都是种子块在候选里排 11~13 名、来回跨 <code>k=10</code> 那条线的用例。
五次的实测幅度：<code>@1</code> ±0.012（1 条），<code>@3</code> ±0.010（1 条），
<code>@10</code> ±0.021（2 条）。<code>@1</code> 前四次纹丝不动，第五次也翻了一条 ——
「某一档是稳定的」不能只靠四个点下结论。</div>

<h2>七、用例明细</h2>
<p>命中位是 <code>@1/@3/@10</code>，<span class="hitmap"><b class="h1">1</b></span> 命中、
<span class="hitmap"><b class="h0">0</b></span> 丢、<span class="hitmap"><b class="hx">-</b></span> 不适用。
名次是种子块在候选里第几 → 在证据里第几，<code>×</code> 表示不在里面。
标红的是 <code>@3</code> 判负的行。「现场」链到导出的 {trace_count} 条 trace。</p>
{table(["用例", "分类", "文档", "@1/@3/@10", "名次", "token", "重复"]
       + (["CR", "CRel"] if judged else []) + [""], case_rows(result["cases"], judged))}

<h2>八、下一步</h2>
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
