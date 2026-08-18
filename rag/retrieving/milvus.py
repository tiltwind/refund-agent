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
from contextlib import contextmanager
from dataclasses import dataclass

from rag.retrieving.pipeline.assemble import assemble
from rag.retrieving.pipeline.recall import recall
from rag.retrieving.pipeline.rerank import MIN_SCORE, rerank
from rag.retrieving.pipeline.rewrite import rewrite
from rag.retrieving.pipeline.route import route
from rag.retrieving.protocol import PolicySection, RetrievalTrace
from rag.retrieving.store import COLLECTION, MILVUS_URI

DEFAULT_TOP_K = int(os.getenv("REFUND_AGENT_POLICY_K", "4"))
"""装配后注入上下文的证据块数（每块是一个回填后的完整小节，不是一个子块）。"""

TRACE = os.getenv("REFUND_AGENT_RAG_TRACE", "").lower() in ("on", "1", "true")
"""把六步打到 stdout。"""

SPAN = os.getenv("REFUND_AGENT_RAG_SPAN", "on").lower() not in ("off", "0", "false")
"""把六步上报 Langfuse。没配密钥时它本来就是空转，这个开关是给「配了密钥但这次
不想让 trace 进库」的场景用的 —— 主要是离线跑批，那种 trace 不挂在任何 dataset run
上，堆在项目里只是噪音。"""


@dataclass
class _Step:
    """一步的产物：`text` 给人读（进 trace、进终端），`data` 给机器读（进 span）。"""

    text: str = ""
    data: dict | None = None


@contextmanager
def _span(name: str, as_type: str = "span", **fields):
    """开一个 span，关掉或没配 Langfuse 时 `yield None`。"""
    if not SPAN:
        yield None
        return

    # 延迟 import：services 是上层，检索链路不该在 import 期就把它拖进来
    from services import telemetry

    with telemetry.span(name, as_type=as_type, input=fields or None) as current:
        yield current


@contextmanager
def _step(trace: RetrievalTrace, name: str, span_name: str, **span_input):
    """一步的收尾动作：把结果同时写进 trace 和 span。

    两个出口记的是同一件事，但不能只留一个：trace 是进程内的，出了这次调用就没了；
    span 进 Langfuse，是事后翻坏 case 用的。抛异常时两个出口都不写 —— 那一步没跑完，
    记一条半截的产物只会误导。

    名字给两个：`name` 是中文步骤名，进 trace 给人读；`span_name` 是英文，Langfuse 上
    要按它跨 trace 筛「所有重排步骤」，跟 langchain 那些 span 的命名也对得上。
    """
    step = _Step()
    with _span(f"rag.{span_name}", **span_input) as current:
        yield step
        trace.record(name, step.text)
        if current is not None:
            current.update(output=step.data if step.data is not None else step.text)


class MilvusRagService:
    def search_policy(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[PolicySection]:
        return self.search_with_trace(query, top_k)[0]

    def search_with_trace(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> tuple[list[PolicySection], RetrievalTrace]:
        """检索评测的入口 —— 除了装配结果，还要中间产物。

        Recall@k 挂在重排输出的 chunk_id 上，而 `search_policy` 只返回装配后的
        `PolicySection`，那上面没有 chunk_id（装配把子块还原成小节，一条对应
        一组子块）。评测脚本自己按六步再编排一遍也能拿到，但那等于把编排逻辑
        抄一份出去，两边迟早跑偏；改成同一段代码多一个出口，prod 与 eval 走的
        就还是同一条链路（2-design 3.4）。
        """
        trace = RetrievalTrace(query=query)

        with _span("rag.search_policy", "retriever", query=query, top_k=top_k) as root:
            # 1 · 改写：拆多意图、判断要不要法规层。失败自动降级为原文透传
            with _step(trace, "改写", "rewrite", query=query) as step:
                plan = rewrite(query)
                step.text = (
                    f"{'LLM' if plan.rewritten else '透传'} → "
                    + "；".join(f"[{q.intent}] {q.text}" for q in plan.sub_queries)
                    + f"（needs_law={plan.needs_law}）"
                )
                step.data = {
                    "rewritten": plan.rewritten,
                    "needs_law": plan.needs_law,
                    "sub_queries": [
                        {"id": q.id, "intent": q.intent, "text": q.text} for q in plan.sub_queries
                    ],
                }

            # 2 · 路由：每条子查询打哪些层、各取多少
            with _step(trace, "路由", "route") as step:
                routes = route(plan)
                step.text = "；".join(
                    f"{r.sub_query.id}→{r.layer_k}(law_w={r.law_weight})" for r in routes
                )
                step.data = {
                    "routes": [
                        {"sub_query": r.sub_query.id, "layer_k": r.layer_k, "law_weight": r.law_weight}
                        for r in routes
                    ]
                }

            # 3 · 过滤在召回内部按层生成（生效日期 + 层级范围，只做硬约束）
            # 4 · 召回融合：每层各跑 dense + BM25 两路，RRF 合并
            with _step(trace, "召回融合", "recall") as step:
                candidates = recall(routes)
                trace.candidate_ids = [c.chunk_id for c in candidates]
                step.text = (
                    f"{len(candidates)} 个候选，"
                    f"单路命中 {sum(1 for c in candidates if c.single_path)} 个；"
                    f"top3={[c.chunk_id for c in candidates[:3]]}"
                )
                # 候选的完整序列，不截断：Recall@10 读的就是它，翻 trace 时也要能看到
                # 「种子块排在第 13 位」这种事 —— 只记 top3 就答不了这个问题
                step.data = {
                    "candidates": trace.candidate_ids,
                    "single_path": sum(1 for c in candidates if c.single_path),
                }

            # 一条都没有：collection 空了 / 灌库没跑 / 条款全被生效日期过滤掉 ——
            # 这是运维故障，不是「没有适用政策」。显式失败，绝不让 Agent 带着一句
            # 「未检索到条款」继续往下判定，那等于把它推回「凭记忆编政策」。
            if not candidates:
                raise RuntimeError(
                    f"policy collection「{COLLECTION}」检索不到任何生效条款"
                    f"（uri={MILVUS_URI}）；请先执行 python rag/index/seed_milvus.py 灌库"
                )

            # 5 · 重排：cross-encoder + 层级/文档先验
            with _step(trace, "重排", "rerank") as step:
                evidence = rerank(query, candidates, routes)
                trace.evidence_ids = [e.row["chunk_id"] for e in evidence]
                step.text = (
                    f"{len(candidates)} → {len(evidence)} 条过阈值；"
                    + "；".join(f"{e.row['chunk_id']}={e.score:.3f}" for e in evidence[:3])
                )
                step.data = {
                    "passed": len(evidence),
                    "min_score": MIN_SCORE,
                    "evidence": [
                        {
                            "chunk_id": e.row["chunk_id"],
                            "score": round(e.score, 3),
                            "relevance": round(e.relevance, 3),
                            "prior": round(e.prior, 2),
                        }
                        for e in evidence
                    ],
                }

            # 6 · 装配：按父块去重、相邻合并、回填原文、预算截断
            with _step(trace, "装配", "assemble") as step:
                sections = assemble(evidence, top_k)
                step.text = f"{len(evidence)} → {len(sections)} 块：" + "；".join(
                    s.section for s in sections
                )
                step.data = {"sections": [s.section for s in sections]}

            if root is not None:
                root.update(output={"sections": [s.section for s in sections]})

        if TRACE:
            print(trace.render())
        return sections, trace
