"""政策检索的唯一实现 —— Agent 直连 Milvus。

**这一层不按 request_source 分实现**：prod 与 eval 走同一条检索路径、同一个
collection。理由与代价见 README 6.4，一句话：评估用的必须是线上真正会返回的
条款，为离线评估另造一份写死的政策，等于把「检索到的条款是否支撑判定」这段
逻辑排除在回归之外，而这恰恰是答复被投诉时最常出问题的一段。

代价是知识库改版会传导到离线回归（同一条用例昨天过今天挂），靠两件事兜住：
collection 按版本发布（MILVUS_COLLECTION 指向固定版本），以及检索结果记进
trace —— 报告波动时先看条款是否变了，再怀疑 Agent。
"""

import os
from datetime import date
from functools import lru_cache

from services.rag.embeddings import build_embeddings
from services.rag.protocol import PolicySection

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "refund_policies")

# 政策条款总共只有个位数条，k 取大一点，一次检索就能覆盖窗口 + 类目 + 商品条件
# 三类要素，从供给侧消掉「逐项各查一次」的必要。
DEFAULT_TOP_K = int(os.getenv("REFUND_AGENT_POLICY_K", "4"))


@lru_cache(maxsize=1)
def _client():
    # 连接和嵌入模型都是重对象，进程内单例复用；首次调用时才建，
    # 让不检索的路径（比如只跑规则引擎的自检）不必依赖 Milvus 在线。
    from pymilvus import MilvusClient

    return MilvusClient(uri=MILVUS_URI)


@lru_cache(maxsize=1)
def _embeddings():
    return build_embeddings()


class MilvusRagService:
    def search_policy(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[PolicySection]:
        today = date.today().isoformat()
        # 生效日期过滤必须走 Milvus 的 filter 表达式 —— 不能把已废止的条款
        # 检索出来再指望模型自己判断哪条还有效（README 第十章）。
        # 日期用 ISO 字符串，字典序即时间序。
        expr = f'effective_date <= "{today}" and expire_date > "{today}"'

        hits = _client().search(
            collection_name=COLLECTION,
            data=[_embeddings().embed_query(query)],
            limit=top_k,
            filter=expr,
            output_fields=["section", "text"],
        )
        rows = hits[0] if hits else []
        # 一条都检索不到，说明 collection 空了 / 灌库没跑 / 条款全被生效日期过滤掉
        # ——这是运维故障，不是「没有适用政策」。显式失败，绝不让 Agent 带着一句
        # 「未检索到条款」继续往下判定，那等于把它推回「凭记忆编政策」。
        if not rows:
            raise RuntimeError(
                f"policy collection「{COLLECTION}」检索不到任何生效条款"
                f"（uri={MILVUS_URI}，生效日期过滤 {today}）；"
                "请先执行 python knowledge/seed_milvus.py 灌库"
            )
        return [
            PolicySection(
                section=h["entity"]["section"],
                text=h["entity"]["text"],
                score=h["distance"],
            )
            for h in rows
        ]
