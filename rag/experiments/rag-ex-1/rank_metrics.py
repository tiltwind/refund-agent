"""三个不调模型的排序指标 —— 判分只读 `source` 与检索链路的 ID 序列。

    candidate_hit  召回层有没有把出题的条文捞进候选集（20 条）
    hit@1 / hit@4  重排后它排在第几
    mrr            1 / 排名，掉出证据列表记 0

## 为什么要有它

两个 LLM 指标在 r1 上已经饱和：非空的 87 条里 84 条 Context Recall 满分，
Context Precision 的位置加权又让尾部的无关条款几乎免费。改一次检索参数，
两个数字都不动 —— 那评测就没有分辨力，后面所有调参结论都无从验证。

这三个指标反过来：不调模型、零方差、跑批之外还能当门禁。更重要的是它们把
「召回层坏了」和「重排/阈值坏了」分开 —— `candidate_hit` 高而 `hit@4` 低，
要改的就不是切片和分析器，是重排权重与 `MIN_SCORE`。

## 它测的是下界，不是 Recall@k

`source` 是**出题用的那一段条文**，不是「这个问题的全部相关块」—— 后者要在
353 个块里人工穷举，漏标无法避免（4 · 一）。所以这里判的是「出题的那一段有没有
被检回来」：它命中不等于检索完备，它没命中则一定漏了。作为下界指标，
它不需要穷举标注就能用，这也是它与 Recall@k 的区别。

## 跨块样本按最深的那个算

跨块样本（14 条）的标准答案要两段条文都用上，只召回一半答不全。所以一条样本的
排名取**全部 `source` 里排得最深的那个**：两段都进前 4 才算 hit@4。单块样本
（82 条）下这个定义与常规写法完全一致。
"""


def rank_metrics(source: list[str], candidate_ids: list[str], evidence_ids: list[str]) -> dict:
    """`source` 缺失时返回空字典 —— 这条样本没有这几个指标，不是判负。"""
    if not source:
        return {}

    rank = _deepest(source, evidence_ids)
    return {
        "candidate_hit": 1.0 if _deepest(source, candidate_ids) else 0.0,
        "hit@1": 1.0 if rank == 1 else 0.0,
        "hit@4": 1.0 if rank and rank <= 4 else 0.0,
        "mrr": round(1.0 / rank, 3) if rank else 0.0,
    }


def _deepest(source: list[str], ranked: list[str]) -> int | None:
    """全部 source 里排得最深的那个的位次（从 1 起）；有一个不在列表里就返回 None。"""
    ranks = []
    for chunk_id in source:
        if chunk_id not in ranked:
            return None
        ranks.append(ranked.index(chunk_id) + 1)
    return max(ranks)


def explain(source: list[str], candidate_ids: list[str], evidence_ids: list[str]) -> str:
    """一行判负理由。报告和 Langfuse 上都要能直接读出「卡在哪一步」。"""
    if not source:
        return "样本没有 source"

    rank = _deepest(source, evidence_ids)
    if rank:
        return f"证据列表第 {rank} 位（共 {len(evidence_ids)} 条）"
    if _deepest(source, candidate_ids):
        pos = max(candidate_ids.index(s) + 1 for s in source)
        return f"候选第 {pos} 位，未进证据列表 —— 卡在重排或 MIN_SCORE"
    return f"不在 {len(candidate_ids)} 条候选里 —— 卡在召回层"
