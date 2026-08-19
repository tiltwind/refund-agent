"""Step 5 · 重排 —— 把「相关」变成「能回答这个问题」。

召回阶段追求**召回率**（宁多勿漏），重排阶段追求**精确率**（把最能回答问题的
顶上去）。两者目标函数不同，所以必须分两段 —— 一段做不了两件事。

## 交叉编码打的是什么分

双塔编码把 query 和条款各自压成一个向量再算距离，判断的是「主题像不像」；
cross-encoder 让两边的 token 互相 attend，判断的是「这段文字回没回答这个问题」。
差别在这个语料上很具体：P02 第二条（退货窗口）和 P07 第三条（会员权益延长）
主题几乎一样，但只有前者能回答「金牌会员签收 10 天还能退吗」。
召回阶段区分不出来，重排能。

## 为什么加权项里没有 freshness

政策是常青内容。一条 2024 年生效、至今未改的核心条款不会因为「旧」而变得
不适用，给它加时间衰减只会让 P02 输给刚发布的边缘规则。「哪一版有效」是
正确性判据，已经由生效日期硬过滤解决了（filters.py）——它不是排序信号，
混进来会同时损害正确性和效果。

取而代之的两个先验都来自文档库自身的规则（doc/policy/README.md）：

- **层级偏好**：答复消费者引用平台层，法规层默认让位（权重由路由给）；
- **文档权威度**：平台层各文件相互冲突时以 P02 为准，其余只作补充。

## 权重没有校准过

`0.80 / 0.20` 是个起点，不是结论。它依赖具体的 reranker、归一化方式和问题
分布，必须在标注集（query → 应召回的 section）上调 —— 调低相关性权重会让
先验压过语义，调高则先验形同虚设。同理，`MIN_SCORE` 也要按问题类型分设：
定义类允许单条权威证据，判定类要求交叉印证。
"""

import os
from dataclasses import dataclass

from llm.rerank import reranker
from rag.retrieving.pipeline.recall import Candidate
from rag.retrieving.pipeline.route import Route

RELEVANCE_WEIGHT = 0.80
PRIOR_WEIGHT = 0.20

MIN_SCORE = float(os.getenv("REFUND_AGENT_POLICY_MIN_SCORE", "0.30"))
"""低于这个分的候选不进上下文。宁可少给两条，也别让弱相关条款进去 ——
模型会认真引用你给它的每一条，包括不该引的那条。"""

DOC_PRIOR = {"P02": 1.0}
"""P02 是退款判定的核心条款，平台层内部冲突时以它为准。其余文档取默认值。"""
DEFAULT_DOC_PRIOR = 0.9


@dataclass
class Evidence:
    candidate: Candidate
    relevance: float
    prior: float
    score: float

    @property
    def row(self) -> dict:
        return self.candidate.row

    def why(self) -> str:
        """相关性理由 —— 线上出坏 case 时区分「召回错了」和「生成错了」的唯一抓手。"""
        return (
            f"相关性 {self.relevance:.3f} × 先验 {self.prior:.2f} → {self.score:.3f}"
            f"；命中 {', '.join(self.candidate.hits)}"
        )


def rerank(query: str, candidates: list[Candidate], routes: list[Route]) -> list[Evidence]:
    if not candidates:
        return []

    law_weight = min((r.law_weight for r in routes), default=0.5)
    relevances = _relevance(query, candidates)

    evidence: list[Evidence] = []
    for candidate, relevance in zip(candidates, relevances):
        row = candidate.row
        layer_prior = 1.0 if row["layer"] == "platform" else law_weight
        prior = layer_prior * DOC_PRIOR.get(row["doc_id"], DEFAULT_DOC_PRIOR)
        score = RELEVANCE_WEIGHT * relevance + PRIOR_WEIGHT * prior
        evidence.append(
            Evidence(candidate=candidate, relevance=relevance, prior=prior, score=score)
        )

    evidence.sort(key=lambda e: e.score, reverse=True)
    eligible = [e for e in evidence if e.score >= MIN_SCORE]
    return _ensure_route_evidence(eligible, routes)


def _ensure_route_evidence(evidence: list[Evidence], routes: list[Route]) -> list[Evidence]:
    """多跳查询先保证每个 rewrite 子查询有一条证据，再按分数补齐。"""
    if len(routes) < 2 or not evidence:
        return evidence

    selected: list[Evidence] = []
    selected_ids: set[str] = set()
    for route in routes:
        prefix = f"{route.sub_query.id}/"
        item = next(
            (e for e in evidence if e.candidate.chunk_id not in selected_ids
             and any(hit.startswith(prefix) for hit in e.candidate.hits)),
            None,
        )
        if item is not None:
            selected.append(item)
            selected_ids.add(item.candidate.chunk_id)
    selected.extend(e for e in evidence if e.candidate.chunk_id not in selected_ids)
    return selected


def _relevance(query: str, candidates: list[Candidate]) -> list[float]:
    """交叉编码分；模型不可用时退回归一化的融合分。

    降级不是等价替换 —— 融合分只知道「两路都排得靠前」，不知道「这段回没回答
    问题」。降级后 P07（会员权益）压过 P02（退货窗口）这类错误就会漏出来。
    所以它是兜底，不是可选配置。
    """
    model = reranker()
    if model is not None:
        # 打分用「块头 + 正文」：块头带着文档标题与条款路径，
        # 对「这段属于哪条规则」的判断贡献很大
        passages = [
            f"{c.row['title']} {c.row['section_path']}\n{c.row['body']}" for c in candidates
        ]
        return model.score(query, passages)

    top = max((c.rrf for c in candidates), default=0.0) or 1.0
    return [c.rrf / top for c in candidates]
