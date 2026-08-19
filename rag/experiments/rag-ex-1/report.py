"""从 result.json 生成 HTML 报告。

    python rag/experiments/rag-ex-1/report.py                      # → report.html
    python rag/experiments/rag-ex-1/report.py --result /tmp/v2.json --out /tmp/v2.html

报告只读结果文件，不连 Langfuse —— 那是本地实例，换台机器 run 页就打不开。

五节：两个 judge 指标、四个排序指标、分数分布、按文档分档、逐条明细。明细里
每条都能展开看 judge 的逐条判定 —— 一个 0.5 分说明不了改哪里，要能翻到是哪一段
被判不相关、哪一句没被支撑；排序那一列直接写出这条卡在召回还是卡在阈值。
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
OUT_PATH = HERE / "report.html"

METRICS = ("context_precision", "context_recall")
RANK_METRICS = ("candidate_hit", "hit@1", "hit@4", "mrr")
LABEL = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "candidate_hit": "候选覆盖",
    "hit@1": "Hit@1",
    "hit@4": "Hit@4",
    "mrr": "MRR",
}

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
main { max-width:1100px; margin:0 auto; }
a { color:var(--series); text-decoration:none; }
a:hover { text-decoration:underline; }
code { font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:18px; margin:48px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--grid); }
h3 { font-size:14px; margin:0 0 10px; font-weight:600; }
p { margin:10px 0; color:var(--ink-2); }
.sub { color:var(--muted); margin:0 0 18px; font-size:14px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
.chip { border:1px solid var(--border); border-radius:999px; padding:3px 11px;
  font-size:12.5px; color:var(--ink-2); background:var(--surface); }
.chip b { color:var(--ink); font-weight:600; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:20px 0 8px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
.tile .k { font-size:12px; color:var(--muted); }
.tile .v { font-size:30px; font-weight:650; letter-spacing:-0.02em; margin-top:2px; }
.tile .n { font-size:12.5px; color:var(--ink-2); }
.formula { margin:14px 0; padding:12px 16px; background:var(--surface); border:1px solid var(--border);
  border-radius:8px; font:12.5px/1.9 ui-monospace,Menlo,monospace; color:var(--ink-2); overflow-x:auto; }
.note { margin:18px 0; padding:14px 16px; background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--warn); border-radius:8px; font-size:14px; color:var(--ink-2); }
.note strong { color:var(--ink); }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px 24px; }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:7px 9px; text-align:left; border-bottom:1px solid var(--grid); white-space:nowrap; }
thead th { font-size:11.5px; color:var(--muted); font-weight:600; background:var(--surface); position:sticky; top:0; }
tbody tr:last-child td { border-bottom:0; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
td.wrap { white-space:normal; min-width:260px; }
.id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.mini { position:relative; height:11px; background:var(--track); border-radius:3px; min-width:60px; }
.mini i { position:absolute; left:0; top:0; height:11px; background:var(--series); border-radius:3px; }
.bar { display:flex; align-items:center; gap:8px; margin:3px 0; font-size:12.5px; color:var(--ink-2); }
.bar .lab { width:78px; font-family:ui-monospace,Menlo,monospace; font-size:12px; text-align:right; }
.bar .track { flex:1; height:14px; background:var(--track); border-radius:3px; position:relative; }
.bar .track i { position:absolute; left:0; top:0; height:14px; background:var(--series); border-radius:3px; }
.bar .cnt { width:34px; font-variant-numeric:tabular-nums; }
details { margin:0; }
details summary { cursor:pointer; color:var(--series); font-size:12px; }
details .detail { margin:8px 0 4px; padding:10px 12px; background:var(--page);
  border:1px solid var(--grid); border-radius:6px; white-space:normal; }
details .detail div { margin:4px 0; font-size:12.5px; color:var(--ink-2); }
.h1 { color:var(--good); font-weight:600; }
.h0 { color:var(--bad); font-weight:600; }
.row-bad td { background:color-mix(in srgb,var(--bad) 6%,transparent); }
footer { margin-top:56px; padding-top:16px; border-top:1px solid var(--grid); font-size:12.5px; color:var(--muted); }
"""


def esc(text) -> str:
    return html.escape(str(text))


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def mini(value) -> str:
    """一格进度条。0/1 的样本一眼看出是满格还是空格。"""
    if value is None:
        return '<span class="id">—</span>'
    return f'<div class="mini"><i style="width:{value * 100:.0f}%"></i></div>'


# ── 各节 ──────────────────────────────────────────────────────────────────
def head(data: dict) -> str:
    cfg = data["config"]
    chips = [
        ("数据集", data["dataset"]),
        ("judge", cfg["judge_model"]),
        ("collection", cfg["collection"]),
        ("top_k", cfg["top_k"]),
        ("min_score", cfg["min_score"]),
        ("耗时", f"{data['elapsed_s']:.0f}s"),
    ]
    run_url = data.get("run_url")
    link = f' · <a href="{esc(run_url)}">Langfuse run</a>' if run_url else ""
    return (
        f"<h1>rag-ex-1 · {esc(data['run_name'])}</h1>"
        f"<p class=\"sub\">{data['summary']['n']} 条样本 · {esc(data['written_at'])}{link}</p>"
        '<div class="chips">'
        + "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
        + "</div>"
    )


def overview(data: dict) -> str:
    s = data["summary"]
    tiles = "".join(
        f'<div class="tile"><div class="k">{LABEL[name]}</div>'
        f'<div class="v">{fmt(s[name])}</div>'
        f'<div class="n">n={s["counted"][name]}</div></div>'
        for name in METRICS
    )
    tiles += (
        f'<div class="tile"><div class="k">空上下文</div>'
        f'<div class="v">{len(s["empty_context"])}</div>'
        f'<div class="n">两个 judge 指标都判 0</div></div>'
    )
    formula = (
        "Context Precision = Σ(Precision@k × rel_k) / 相关条数<br>"
        "&nbsp;&nbsp;逐段判这段条款对回答问题有没有用，按位置加权：相关的排在前面得分高<br><br>"
        "Context Recall&nbsp;&nbsp;&nbsp;&nbsp;= 被上下文支撑的句子数 / ground_truth 的句子数<br>"
        "&nbsp;&nbsp;标准答案按句拆开，逐句判检回的上下文里有没有这个信息"
    )
    warn = ""
    if s["judge_errors"]:
        warn = (
            f'<div class="note"><strong>judge 失败 {len(s["judge_errors"])} 次</strong>，'
            "这些条目没写分数、不在均值的分母里："
            + esc("；".join(s["judge_errors"][:5]))
            + "</div>"
        )
    if s.get("failures"):
        warn += (
            f'<div class="note"><strong>跑批故障 {len(s["failures"])} 条</strong>，'
            "这些条目一个分数都没写、不在任何均值的分母里（故障不记到检索头上）："
            + esc("；".join(s["failures"][:5]))
            + "</div>"
        )
    return f'<h2>一、两个 judge 指标</h2><div class="formula">{formula}</div><div class="tiles">{tiles}</div>{warn}'


def ranking(data: dict) -> str:
    """排序指标 —— 不调模型，只比对 `source` 与检索链路的 ID 序列。"""
    s = data["summary"]
    if s.get("hit@4") is None:
        return ""
    tiles = "".join(
        f'<div class="tile"><div class="k">{LABEL[name]}</div>'
        f'<div class="v">{fmt(s[name])}</div>'
        f'<div class="n">n={s["counted"][name]}</div></div>'
        for name in RANK_METRICS
    )
    stuck = [
        c for c in data["cases"]
        if c["scores"].get("candidate_hit") == 1.0 and c["scores"].get("hit@4") == 0.0
    ]
    note = (
        '<div class="note"><strong>候选覆盖高、Hit@4 低</strong>的样本有 '
        f"{len(stuck)} 条：出题的条文被召回层捞进来了，却没排进前 4 —— "
        "要动的是重排权重与 <code>MIN_SCORE</code>，不是切片与分析器。</div>"
        if stuck else ""
    )
    return (
        "<h2>二、排序指标</h2>"
        "<p><code>source</code> 是出题用的那一段条文，判的是它有没有被检回来："
        "命中不代表检索完备，没命中则一定漏了 —— 它是下界指标，不需要穷举标注。"
        "跨块样本按两段里排得最深的那个算。</p>"
        f'<div class="tiles">{tiles}</div>{note}'
    )


def distribution(data: dict) -> str:
    """两个指标各一张分布图。均值掩盖形状：0.8 可能是「都在 0.8」，
    也可能是「一半 1.0 一半 0.6」，改参数要救的是后者的下半截。"""
    buckets = [(0.0, 0.001, "= 0"), (0.001, 0.34, "(0, 0.33]"), (0.34, 0.67, "(0.33, 0.67]"),
               (0.67, 0.999, "(0.67, 1)"), (0.999, 1.01, "= 1")]
    blocks = []
    for name in METRICS:
        values = [c["scores"][name] for c in data["cases"] if name in c["scores"]]
        counts = [sum(1 for v in values if lo <= v < hi) for lo, hi, _ in buckets]
        top = max(counts) or 1
        bars = "".join(
            f'<div class="bar"><span class="lab">{esc(label)}</span>'
            f'<span class="track"><i style="width:{n / top * 100:.0f}%"></i></span>'
            f'<span class="cnt">{n}</span></div>'
            for (_, _, label), n in zip(buckets, counts)
        )
        blocks.append(f"<div><h3>{LABEL[name]}</h3>{bars}</div>")
    return '<h2>三、分数分布</h2><div class="grid2">' + "".join(blocks) + "</div>"


def by_doc(data: dict) -> str:
    columns = METRICS + ("hit@4", "mrr")
    rows = "".join(
        f'<tr><td class="id">{esc(doc)}</td><td class="n">{stats["n"]}</td>'
        + "".join(
            f'<td class="n">{fmt(stats.get(name))}</td><td>{mini(stats.get(name))}</td>'
            for name in columns
        )
        + "</tr>"
        for doc, stats in sorted(
            data["breakdown"].items(), key=lambda kv: (kv[1].get("hit@4") or 0, kv[1].get("mrr") or 0)
        )
    )
    return (
        "<h2>四、按文档分档</h2>"
        "<p>样本出自哪篇文档，按 Hit@4 升序。总均值只用来看趋势，能指向改哪一篇语料的"
        "是这张表 —— 某一篇的 Hit@4 明显低于其他，先去看那篇的切片和小节标题。</p>"
        '<div class="tablewrap"><table><thead><tr><th>文档</th><th class="n">n</th>'
        '<th class="n">Precision</th><th></th><th class="n">Recall</th><th></th>'
        '<th class="n">Hit@4</th><th></th><th class="n">MRR</th><th></th>'
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


def _verdicts(row: dict, name: str) -> str:
    """一条样本在一个指标上的逐条判定。"""
    result = (row.get("judge") or {}).get(name) or {}
    if not result.get("detail"):
        return f'<div>{esc(result.get("skipped") or result.get("error") or "无判定")}</div>'
    items = "".join(
        f'<div><span class="{"h1" if d["hit"] else "h0"}">{"✓" if d["hit"] else "✗"}</span> '
        f'{esc(d["text"])}<br><span class="id">{esc(d["reason"])}</span></div>'
        for d in result["detail"]
    )
    return f"<div><b>{LABEL[name]}</b> {result.get('hit')}/{result.get('n')}</div>{items}"


def cases(data: dict) -> str:
    rows = []
    for row in sorted(data["cases"], key=lambda r: (r["scores"].get("hit@4") or 0,
                                                    r["scores"].get("mrr") or 0,
                                                    r["scores"].get("context_precision") or 0)):
        scores = row["scores"]
        bad = any((scores.get(name) or 0) < 1.0 for name in METRICS + ("hit@4",))
        detail = "".join(f'<div class="detail">{_verdicts(row, name)}</div>' for name in METRICS)
        # 排名一列写的是「卡在哪一步」，不是数字：一个 hit@4=0 说明不了要改召回还是改阈值
        note = (row.get("retrieval") or {}).get("rank_note", "")
        rows.append(
            f'<tr class="{"row-bad" if bad else ""}">'
            f'<td class="id">{esc(row["case_id"])}</td>'
            f'<td class="id">{esc(row["doc_id"])}</td>'
            f'<td class="wrap">{esc(row["question"])}'
            f"<details><summary>判定明细</summary>{detail}</details></td>"
            f'<td class="n">{len(row["sections"])}</td>'
            + "".join(f'<td class="n">{fmt(scores.get(name))}</td>' for name in METRICS)
            + f'<td class="n">{fmt(scores.get("mrr"))}</td>'
            + f'<td class="wrap">{esc(note)}</td>'
            + f'<td class="n">{row["elapsed_s"]:.1f}s</td></tr>'
        )
    return (
        f"<h2>五、逐条明细（{len(rows)} 条，按 Hit@4 升序）</h2>"
        '<div class="tablewrap"><table><thead><tr><th>用例</th><th>出处</th><th>问题</th>'
        '<th class="n">段数</th><th class="n">Precision</th><th class="n">Recall</th>'
        '<th class="n">MRR</th><th>source 排名</th>'
        '<th class="n">耗时</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render(data: dict) -> str:
    cfg = " · ".join(f"{k}={v}" for k, v in data["config"].items())
    return (
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>rag-ex-1 · {esc(data['run_name'])}</title><style>{CSS}</style></head><body><main>"
        + head(data)
        + overview(data)
        + ranking(data)
        + distribution(data)
        + by_doc(data)
        + cases(data)
        + f"<footer>被测参数：{esc(cfg)}</footer>"
        + "</main></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从 result.json 生成 HTML 报告")
    parser.add_argument("--result", default=str(RESULT_PATH), help="结果文件")
    parser.add_argument("--out", default=str(OUT_PATH), help="输出的 HTML")
    args = parser.parse_args()

    result = Path(args.result)
    if not result.exists():
        raise SystemExit(f"没有 {result}，先跑 run_experiment.py")

    out = Path(args.out)
    out.write_text(render(json.loads(result.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
