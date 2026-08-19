"""把 doc/policy/ 下的政策文档灌进 Milvus。

    bash scripts/milvus.sh start          # 见 doc/platform/milvus.md
    python rag/index/seed_milvus.py       # 建 collection + 切片 + 灌库

语料源就是 `doc/policy/**/*.md` 本身 —— **没有中间产物**。以前这里读的是一份
手抄的 rag/policies.json，那等于给同一套政策留了两份事实源：文档改了、
JSON 没改（或反之），Agent 就会引用一条与线上公示规则不一致的条款，而且不报错。
现在文档即语料，切片规则见 rag/chunking/policy.py。

重复执行是安全的：默认 drop 后重建。条款是全量小语料（数百个块），增量更新
省不了多少时间，而「旧版本块残留在库里」会让检索同时召回新旧两版，代价大得多。

线上不该这么灌 —— 政策按版本发布、灰度切换 collection，别在同一个 collection
上原地 drop：检索会在重建的空窗期返回空，Agent 直接失败。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymilvus import DataType, Function, FunctionType, MilvusClient  # noqa: E402

from llm.embedding.bge_m3 import DIMENSION, MAX_LENGTH, embedder  # noqa: E402
from rag.chunking import CHILD_MAX_TOKENS, Chunk, chunk_document  # noqa: E402
from rag.retrieving import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "doc" / "policy"
INSERT_BATCH = 64


# ── 切片 ──────────────────────────────────────────────────────────────────
def build_chunks() -> list[Chunk]:
    model = embedder()
    chunks: list[Chunk] = []
    for path in sorted(POLICY_DIR.rglob("*.md")):
        # README.md 是文档库的说明书，不是政策条文；灌进去只会在检索里
        # 和真正的条款抢名额
        if path.name == "README.md":
            continue
        got = chunk_document(path, ROOT, model.count_tokens, model.encode_documents)
        print(f"  {path.relative_to(ROOT)}：{len(got)} 块 / {len({c.parent_id for c in got})} 父块")
        chunks.extend(got)
    return chunks


def check_truncation(chunks: list[Chunk]) -> None:
    """入库前把「静默截断」变成显式失败。

    超过 max_seq_length 的部分**从未进入模型**，对向量的贡献严格为 0。
    这种块躺在库里看起来「已经索引了」，但当答案落在被截掉的后半段时，
    它的向量里没有任何相关信号，永远不会被召回 —— 没有异常、没有日志，
    只会观察到「RAG 效果不好」却定位不到原因。所以这里必须硬卡。
    """
    report = embedder().truncation_report([c.text for c in chunks])
    print(
        f"\ntoken 长度分布：p50={report['p50']} p99={report['p99']} max={report['max']} "
        f"（模型截断上限 {report['limit']}，切分硬上限 {CHILD_MAX_TOKENS}）"
    )
    if report["truncated"]:
        raise SystemExit(
            f"有 {report['truncated']}/{report['n']} 个块超过 {MAX_LENGTH} token 会被静默截断。"
            "调小 POLICY_CHUNK_MAX，或检查是否有超长表格未被拆开。"
        )


# ── 建表 ──────────────────────────────────────────────────────────────────
def create_collection(client: MilvusClient) -> None:
    name = store.COLLECTION
    if client.has_collection(name):
        print(f"drop 已存在的 collection：{name}")
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=96)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("parent_seq", DataType.INT64)

    # 正文与可检索文本分开存：body 是原文（装配后喂模型），
    # text = 块头 + body（进 embedding 与 BM25）。分开之后模型读到的上下文里
    # 没有【文档】【路径】这些检索用的标记。
    schema.add_field("body", DataType.VARCHAR, max_length=16384)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=20480,
        enable_analyzer=True,
        # 中文必须显式指定分析器：默认的 standard 按空白切词，
        # 「金牌会员15天未拆封」会整串变成一个 term，BM25 直接失效
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("kind", DataType.VARCHAR, max_length=16)

    # ── 过滤与重排用的标量字段 ──
    schema.add_field("doc_id", DataType.VARCHAR, max_length=16)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("section_path", DataType.VARCHAR, max_length=512)
    schema.add_field("layer", DataType.VARCHAR, max_length=16)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("authority", DataType.VARCHAR, max_length=256)
    schema.add_field("authority_level", DataType.INT64)
    # 日期用 ISO 字符串：字典序即时间序，能直接进 filter 表达式做比较
    schema.add_field("effective_date", DataType.VARCHAR, max_length=10)
    schema.add_field("expire_date", DataType.VARCHAR, max_length=10)
    # max_length 的单位是**字节**不是字符，中文一字 3 字节。法规层的 version
    # 取自 frontmatter 的 revision，是「1993年通过，2013年第二次修正」这样的整句
    schema.add_field("version", DataType.VARCHAR, max_length=512)
    schema.add_field("tags", DataType.VARCHAR, max_length=512)
    schema.add_field("source_path", DataType.VARCHAR, max_length=256)

    # ── 两路向量 ──
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 由 Milvus 自己算：插入时只给 text，sparse 由 Function 生成，
    # 检索时传查询原文（不是向量）。把词表和 IDF 留在库里，
    # 应用侧就不必维护一份会与 collection 漂移的倒排索引。
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )

    index = client.prepare_index_params()
    # 几百个块，FLAT 就是精确检索 —— 没有 nlist / ef 可抖，
    # 也就少了一个污染评估结论的变量
    index.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(name, schema=schema, index_params=index)
    print(f"\n建表完成：{name}（dense={DIMENSION}d/COSINE/FLAT，sparse=BM25/中文分析器）")


# ── 灌库 ──────────────────────────────────────────────────────────────────
def insert(client: MilvusClient, chunks: list[Chunk]) -> None:
    model = embedder()
    for start in range(0, len(chunks), INSERT_BATCH):
        batch = chunks[start : start + INSERT_BATCH]
        vectors = model.encode_documents([c.text for c in batch])
        client.insert(
            collection_name=store.COLLECTION,
            data=[
                {
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "chunk_index": c.chunk_index,
                    "parent_seq": c.parent_seq,
                    "body": c.body,
                    "text": c.text,
                    "kind": c.kind,
                    "doc_id": c.doc.doc_id,
                    "title": c.doc.title,
                    "section_path": " > ".join(c.section_path),
                    "layer": c.doc.layer,
                    "doc_type": c.doc.doc_type,
                    "category": c.doc.category,
                    "authority": c.doc.authority,
                    "authority_level": c.doc.authority_level,
                    "effective_date": c.doc.effective_date,
                    "expire_date": c.doc.expire_date,
                    "version": c.doc.version,
                    "tags": "、".join(c.doc.tags),
                    "source_path": c.doc.source_path,
                    "dense": v,
                }
                for c, v in zip(batch, vectors)
            ],
        )
        print(f"  已灌 {min(start + INSERT_BATCH, len(chunks))}/{len(chunks)}")


def main() -> None:
    print(f"切片 {POLICY_DIR.relative_to(ROOT)}：")
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"{POLICY_DIR} 下没有可切片的政策文档")
    check_truncation(chunks)

    client = MilvusClient(uri=store.MILVUS_URI)
    create_collection(client)

    print(f"\n灌库 {len(chunks)} 块：")
    insert(client, chunks)
    client.flush(store.COLLECTION)
    client.load_collection(store.COLLECTION)

    docs = sorted({c.doc.doc_id for c in chunks})
    print(
        f"\n完成：{len(chunks)} 个子块 / {len({c.parent_id for c in chunks})} 个父块 / "
        f"{len(docs)} 篇文档 {docs}"
    )


if __name__ == "__main__":
    main()
