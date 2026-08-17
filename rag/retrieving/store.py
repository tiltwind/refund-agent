"""Milvus 存取的薄封装 —— collection 常量、连接单例、三种取数方式。

放在这里而不是散落在 pipeline/ 各步里，是为了让「collection 长什么样」
只有一个定义点：灌库脚本（rag/index/seed_milvus.py）和检索链路都从这里拿
COLLECTION 与字段列表，改字段时不会漏改一边。
"""

import os
from functools import lru_cache

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

COLLECTION = os.getenv("MILVUS_COLLECTION", "refund_policy_chunks")
"""政策块 collection。

**评估环境要把它固定到一个明确的版本**（如 `refund_policy_chunks_v1`）：
政策库换了内容而基线没重跑，回归报告里的涨跌就不再只反映 Agent 的改动。
"""

FIELDS = [
    "chunk_id",
    "parent_id",
    "chunk_index",
    "parent_seq",
    "body",
    "kind",
    "doc_id",
    "title",
    "section_path",
    "layer",
    "doc_type",
    "authority",
    "authority_level",
    "effective_date",
    "expire_date",
    "version",
    "source_path",
]
"""检索要带回的字段。**不含 `text`** —— 那是「块头 + 正文」的索引用文本，
喂给模型的是 `body`。也不含向量，1024 维乘以候选数纯属浪费带宽。"""


@lru_cache(maxsize=1)
def client():
    """连接单例。首次调用时才建，让不检索的路径不必依赖 Milvus 在线。"""
    from pymilvus import MilvusClient

    return MilvusClient(uri=MILVUS_URI)


def search_dense(vector: list[float], expr: str, limit: int) -> list[dict]:
    """稠密召回：BGE-M3 向量的余弦近邻。"""
    hits = client().search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="dense",
        limit=limit,
        filter=expr,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=FIELDS,
    )
    return _rows(hits)


def search_bm25(query: str, expr: str, limit: int) -> list[dict]:
    """稀疏召回：BM25 字面匹配。

    注意传进去的是**查询原文**不是向量 —— 分词与 term 权重由 Milvus 的
    BM25 Function 在服务端算，与灌库时用的是同一套分析器和同一份 IDF 统计。
    这正是把 BM25 放在库里而不是应用层的理由：应用层那份倒排索引迟早会与
    collection 漂移，而漂移不会报错。
    """
    hits = client().search(
        collection_name=COLLECTION,
        data=[query],
        anns_field="sparse",
        limit=limit,
        filter=expr,
        search_params={"metric_type": "BM25", "params": {}},
        output_fields=FIELDS,
    )
    return _rows(hits)


def fetch_parent(parent_id: str) -> list[dict]:
    """取一个父块的全部子块，按 chunk_index 升序 —— 装配时拼回原文用。

    父块不单独存储，它就是这些子块的有序拼接（rag/chunking/model.py）。
    """
    rows = client().query(
        collection_name=COLLECTION,
        filter=f'parent_id == "{parent_id}"',
        output_fields=FIELDS,
        limit=64,
    )
    return sorted(rows, key=lambda r: r["chunk_index"])


def _rows(hits) -> list[dict]:
    """把 Milvus 的 (entity, distance) 结构摊平成一层 dict，附带原始分。"""
    out = []
    for hit in (hits[0] if hits else []):
        row = dict(hit.get("entity") or {})
        row["score"] = float(hit["distance"])
        out.append(row)
    return out
