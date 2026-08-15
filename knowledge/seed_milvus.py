"""把 knowledge/policies.json 灌进 Milvus 的 policy collection。

    bash standalone_embed.sh start        # 见 doc/rag/milvus-service.md
    python knowledge/seed_milvus.py       # 建 collection + 灌条款

重复执行是安全的：默认 drop 后重建（条款是全量小语料，增量意义不大，
而「旧版本条款残留在库里」会让检索同时召回新旧两版，比重灌代价大得多）。

线上不该这么灌 —— 政策条款按版本发布、灰度切换 collection，别在同一个
collection 上原地 drop：检索会在重建的空窗期返回空，Agent 直接失败。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import DataType, MilvusClient  # noqa: E402

from services.rag import milvus as rag_milvus  # noqa: E402
from services.rag.embeddings import build_embeddings, dimension  # noqa: E402

SOURCE = Path(__file__).parent / "policies.json"


def main() -> None:
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    sections = doc["sections"]

    embeddings = build_embeddings()
    dim = dimension(embeddings)

    client = MilvusClient(uri=rag_milvus.MILVUS_URI)
    collection = rag_milvus.COLLECTION

    if client.has_collection(collection):
        print(f"drop 已存在的 collection：{collection}")
        client.drop_collection(collection)

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("section", DataType.VARCHAR, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    # 生效日期做成标量字段，检索时用 filter 表达式排除已废止版本 ——
    # 不能指望模型自己从检索结果里判断哪条还有效（README 第十章）
    schema.add_field("effective_date", DataType.VARCHAR, max_length=10)
    schema.add_field("expire_date", DataType.VARCHAR, max_length=10)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    index = client.prepare_index_params()
    # 条款只有个位数条，FLAT 就是精确检索，没有召回抖动 ——
    # 索引参数（nlist / ef）本身也是评估结论的污染源，这里直接消掉它
    index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")

    client.create_collection(collection, schema=schema, index_params=index)
    print(f"建表完成：{collection}（dim={dim}, metric=COSINE, index=FLAT）")

    vectors = embeddings.embed_documents([s["text"] for s in sections])
    client.insert(
        collection_name=collection,
        data=[
            {
                "section": s["section"],
                "text": s["text"],
                "effective_date": s["effective_date"],
                "expire_date": s["expire_date"],
                "vector": v,
            }
            for s, v in zip(sections, vectors)
        ],
    )
    client.flush(collection)
    print(f"灌入 {len(sections)} 条条款：{[s['section'] for s in sections]}")


if __name__ == "__main__":
    main()
