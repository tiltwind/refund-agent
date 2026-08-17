"""政策检索的唯一实现 —— Agent 直连 Milvus，六步链路。

    改写 → 路由 → 过滤 → 召回融合 → 重排 → 装配

每一步在 pipeline/ 下有独立模块，本文件只负责编排和落 trace。
拆开的理由与各步的取舍见 rag/retrieving/pipeline/__init__.py。

**这一层不按 request_source 分实现**：prod 与 eval 走同一条检索路径、同一个
collection。理由与代价见 2-design 3.4，一句话：评估用的必须是线上真正会返回的
条款，为离线评估另造一份写死的政策，等于把「检索到的条款是否支撑判定」这段
逻辑排除在回归之外，而这恰恰是答复被投诉时最常出问题的一段。

代价是知识库改版会传导到离线回归（同一条用例昨天过今天挂），靠两件事兜住：
collection 按版本发布（MILVUS_COLLECTION 指向固定版本），以及检索链路每一步
的中间产物记进 trace —— 报告波动时先看条款是否变了，再怀疑 Agent。
"""

import os

from rag.retrieving.pipeline.assemble import assemble
from rag.retrieving.pipeline.recall import recall
from rag.retrieving.pipeline.rerank import rerank
from rag.retrieving.pipeline.rewrite import rewrite
from rag.retrieving.pipeline.route import route
from rag.retrieving.protocol import PolicySection, RetrievalTrace
from rag.retrieving.store import COLLECTION, MILVUS_URI

DEFAULT_TOP_K = int(os.getenv("REFUND_AGENT_POLICY_K", "4"))
"""装配后注入上下文的证据块数（每块是一个回填后的完整小节，不是一个子块）。"""

TRACE = os.getenv("REFUND_AGENT_RAG_TRACE", "").lower() in ("on", "1", "true")


class MilvusRagService:
    def search_policy(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[PolicySection]:
        trace = RetrievalTrace(query=query)

        # 1 · 改写：拆多意图、判断要不要法规层。失败自动降级为原文透传
        plan = rewrite(query)
        trace.record(
            "改写",
            f"{'LLM' if plan.rewritten else '透传'} → "
            + "；".join(f"[{q.intent}] {q.text}" for q in plan.sub_queries)
            + f"（needs_law={plan.needs_law}）",
        )

        # 2 · 路由：每条子查询打哪些层、各取多少
        routes = route(plan)
        trace.record(
            "路由",
            "；".join(f"{r.sub_query.id}→{r.layer_k}(law_w={r.law_weight})" for r in routes),
        )

        # 3 · 过滤在召回内部按层生成（生效日期 + 层级范围，只做硬约束）
        # 4 · 召回融合：每层各跑 dense + BM25 两路，RRF 合并
        candidates = recall(routes)
        trace.record(
            "召回融合",
            f"{len(candidates)} 个候选，"
            f"单路命中 {sum(1 for c in candidates if c.single_path)} 个；"
            f"top3={[c.chunk_id for c in candidates[:3]]}",
        )

        # 一条都没有：collection 空了 / 灌库没跑 / 条款全被生效日期过滤掉 ——
        # 这是运维故障，不是「没有适用政策」。显式失败，绝不让 Agent 带着一句
        # 「未检索到条款」继续往下判定，那等于把它推回「凭记忆编政策」。
        if not candidates:
            raise RuntimeError(
                f"policy collection「{COLLECTION}」检索不到任何生效条款"
                f"（uri={MILVUS_URI}）；请先执行 python rag/index/seed_milvus.py 灌库"
            )

        # 5 · 重排：cross-encoder + 层级/文档先验
        evidence = rerank(query, candidates, routes)
        trace.record(
            "重排",
            f"{len(candidates)} → {len(evidence)} 条过阈值；"
            + "；".join(f"{e.row['chunk_id']}={e.score:.3f}" for e in evidence[:3]),
        )

        # 6 · 装配：按父块去重、相邻合并、回填原文、预算截断
        sections = assemble(evidence, top_k)
        trace.record(
            "装配",
            f"{len(evidence)} → {len(sections)} 块："
            + "；".join(s.section for s in sections),
        )

        if TRACE:
            print(trace.render())
        return sections
