# RAG 检索退款政策：核心代码

政策条款存进 Milvus，两条独立流程共用同一个 collection：切片建库、六步检索。切片把政策 Markdown 变成父子块；检索链路依次是改写、路由、过滤、召回融合、重排、装配。检索不到证据时抛异常，不返回空结果。

## 一、切片：父子块与块头

子块是唯一入库的单元，`chunk_id` 由 `parent_id:序号` 拼成；父块不单独存储，检索时按 `parent_id` 把同组子块拼接还原。

```python
@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    chunk_index: int          # 父块内的序号，回填时按它排序拼接
    doc: DocMeta
    section_path: tuple[str, ...]
    body: str                 # 原文，喂给模型的就是它
    kind: str = "text"        # text | table | code，表格代码不再切分
    parent_seq: int = 0       # 父块顺序号，判断两个父块是否相邻

    @property
    def header(self) -> str:
        lines = [f"【文档】{self.doc.doc_id} {self.doc.title}"]
        if self.section_path:
            lines.append(f"【路径】{' > '.join(self.section_path)}")
        return "\n".join(lines)

    @property
    def text(self) -> str:    # 块头 + 正文，进 embedding 与 BM25
        return f"{self.header}\n\n{self.body}"
```

父块按顶层标题（`##`）分组，子块在父块内按段落打包到目标 320 / 硬上限 512 token，overlap 恒为 0：

```python
def chunk_document(path, root, tokens, encode) -> list[Chunk]:
    rel = str(path.relative_to(root))
    doc, body = parse_frontmatter(path.read_text(encoding="utf-8"), rel)

    chunks = []
    for seq, group in enumerate(_group_by_top_heading(split_sections(body))):
        parent_id = f"{doc.doc_id}#{seq:03d}"
        idx = 0
        for section in group:
            for block in _pack(split_blocks(section.text), tokens, encode):
                chunks.append(Chunk(
                    chunk_id=f"{parent_id}:{idx:02d}", parent_id=parent_id,
                    chunk_index=idx, doc=doc, section_path=section.path,
                    body=block.text, kind=block.kind, parent_seq=seq,
                ))
                idx += 1
    return chunks


def _pack(blocks, tokens, encode) -> list[Block]:
    out, cur, cur_tokens = [], [], 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            out.append(_merge(cur))
            cur, cur_tokens = [], 0

    for block in blocks:
        n = tokens(block.text)
        if block.atomic:                       # 表格 / 代码原子块，永不切开
            if cur and cur_tokens + n > CHILD_MAX_TOKENS:
                flush()
            cur.append(block); cur_tokens += n
            if cur_tokens >= CHILD_TARGET_TOKENS:
                flush()
            continue
        if n > CHILD_MAX_TOKENS:               # 超长自然段交给语义切分兜底
            flush()
            out.extend(Block("text", p) for p in
                       semantic_split(block.text, encode, tokens,
                                      CHILD_TARGET_TOKENS, CHILD_MAX_TOKENS))
            continue
        if cur and cur_tokens + n > CHILD_TARGET_TOKENS:
            flush()
        cur.append(block); cur_tokens += n

    flush()
    return out
```

## 二、建表与灌库

正文与检索文本分开存：`body` 是原文，`text` 是块头 + 正文，进 embedding 与 BM25。中文分词必须显式指定分析器。

```python
schema.add_field("body", DataType.VARCHAR, max_length=16384)
schema.add_field("text", DataType.VARCHAR, max_length=20480,
                 enable_analyzer=True,
                 analyzer_params={"type": "chinese"})

# BM25 由 Milvus 服务端算，插入只给 text，sparse 由 Function 生成
schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))

schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

index = client.prepare_index_params()
index.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

`effective_date`、`expire_date`、`layer`、`authority_level` 等标量字段单独存，供后续硬过滤与重排加权，不进向量：

```python
def insert(client, chunks):
    model = embedder()
    for start in range(0, len(chunks), INSERT_BATCH):
        batch = chunks[start : start + INSERT_BATCH]
        vectors = model.encode_documents([c.text for c in batch])
        client.insert(collection_name=store.COLLECTION, data=[
            {
                "chunk_id": c.chunk_id, "parent_id": c.parent_id,
                "body": c.body, "text": c.text, "dense": v,
                "doc_id": c.doc.doc_id, "layer": c.doc.layer,
                "effective_date": c.doc.effective_date,
                "expire_date": c.doc.expire_date,
                "authority_level": c.doc.authority_level,
                # 其余标量字段同理，逐个对应 DocMeta 的字段
            }
            for c, v in zip(batch, vectors)
        ])
```

## 三、检索链路总览

```python
def search_policy(query: str, top_k: int = 4) -> list[PolicySection]:
    plan = rewrite(query)
    routes = route(plan)
    candidates = recall(routes)
    if not candidates:
        raise NoCandidatesError(f"检索不到任何生效条款：{query}")

    evidence = rerank(query, candidates, routes)
    sections = assemble(evidence, top_k)
    if not sections:
        raise NoEvidenceError(f"重排后没有可交付证据：{query}")
    return sections
```

## 四、改写：拆多意图、判断法规层

改写的输出是完整的自然语言问句，不是关键词串——重排用的 cross-encoder 判断的是"这段文字回没回答这个问题"，关键词串没有疑问点，排序会退化成主题相似度。生效日期不经过改写，由过滤那一步用 `date.today()` 计算。

```python
class SubQuery(BaseModel):
    id: str
    intent: Intent
    text: str          # 完整自然语言问句，带上类目、天数、会员等级等已知条件

class Rewrite(BaseModel):
    sub_queries: list[SubQuery]
    needs_law: bool     # 是否需要召回法规层

def rewrite(query: str) -> RetrievalPlan:
    if not ENABLED or not chat.available():
        return _passthrough(query)
    try:
        result = _model().invoke([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": query},
        ])
    except Exception:
        return _passthrough(query)     # 改写失败，原文透传，检索照常进行

    return RetrievalPlan(
        original=query, sub_queries=result.sub_queries[:3],
        needs_law=result.needs_law, rewritten=True,
    )
```

## 五、路由：平台层与法规层各给多少名额

`platform` 是平台与消费者的直接约定，答复引用的就是它；`law` 是法定底线，用来判断平台条款是否有效。法规层从不被硬砍掉，路由只调节名额与后续重排权重。

```python
PLATFORM_K, LAW_K = 18, 5
VALIDITY_PLATFORM_K = VALIDITY_LAW_K = 12
LAW_INTENTS = {"validity", "dispute"}

def route(plan: RetrievalPlan) -> list[Route]:
    routes = []
    for sub in plan.sub_queries:
        law_first = plan.needs_law or sub.intent in LAW_INTENTS
        if law_first:
            routes.append(Route(sub, {"platform": VALIDITY_PLATFORM_K, "law": VALIDITY_LAW_K},
                                 law_weight=1.0))
        else:
            routes.append(Route(sub, {"platform": PLATFORM_K, "law": LAW_K},
                                 law_weight=0.5))    # 法规层仍召回，排序上让位
    return routes
```

## 六、过滤：生效日期与层级，只做硬约束

时间不参与排序，只做硬过滤：政策是常青内容，不因为"旧"而失效，过滤只负责把已废止的版本挡在候选池外。

```python
def build_filter(layers: list[str], today: str | None = None) -> str:
    day = today or date.today().isoformat()
    layer_list = ", ".join(f'"{x}"' for x in layers)
    return (
        f'effective_date <= "{day}" and expire_date > "{day}" '
        f"and layer in [{layer_list}]"
    )
```

## 七、召回融合：稠密 + BM25 双路，RRF 合并

BM25 精确命中条款号、天数、次数这类低频高权重 term；稠密向量补上措辞不重合的语义命中。两路各自跑出排名列表后在应用层做 RRF，只用排名、不比较原始分数，绕开两路分数尺度不可比的问题。

```python
RRF_K = 20
CANDIDATE_LIMIT = 20

def recall(routes: list[Route]) -> list[Candidate]:
    pool: dict[str, Candidate] = {}
    model = embedder()

    for route_ in routes:
        query = route_.sub_query.text
        vector = model.encode_query(query)
        for layer, k in route_.layer_k.items():
            expr = build_filter([layer])          # 按层分别检索，各层单独有名额
            tag = f"{route_.sub_query.id}/{layer}"
            dense = store.search_dense(vector, expr, k)
            bm25 = store.search_bm25(query, expr, k)
            _fuse(pool, dense, f"{tag}/dense")
            _fuse(pool, bm25, f"{tag}/bm25")

    ranked = sorted(pool.values(), key=lambda c: c.rrf, reverse=True)
    return ranked[:CANDIDATE_LIMIT]


def _fuse(pool: dict, rows: list[dict], tag: str) -> None:
    for rank, row in enumerate(rows, start=1):
        candidate = pool.get(row["chunk_id"])
        if candidate is None:
            candidate = pool[row["chunk_id"]] = Candidate(row=row)
        candidate.rrf += 1.0 / (RRF_K + rank)
        candidate.hits.append(f"{tag}#{rank}")
```

## 八、重排：交叉编码 + 层级 / 文档先验

召回追求召回率，重排追求精确率：交叉编码让 query 与条款的 token 互相 attend，判断的是"这段回没回答问题"，而不是双塔向量的"主题像不像"。最终分是相关性与先验的加权和，先验只含层级权重和文档权威度两项，不掺时间衰减。

```python
RELEVANCE_WEIGHT, PRIOR_WEIGHT = 0.80, 0.20
MIN_SCORE = 0.30
DOC_PRIOR = {"P02": 1.0}       # 平台层内部冲突时以 P02 为准
DEFAULT_DOC_PRIOR = 0.9

def rerank(query: str, candidates: list[Candidate], routes: list[Route]) -> list[Evidence]:
    law_weight = min((r.law_weight for r in routes), default=0.5)
    relevances = _relevance(query, candidates)

    evidence = []
    for candidate, relevance in zip(candidates, relevances):
        row = candidate.row
        layer_prior = 1.0 if row["layer"] == "platform" else law_weight
        prior = layer_prior * DOC_PRIOR.get(row["doc_id"], DEFAULT_DOC_PRIOR)
        score = RELEVANCE_WEIGHT * relevance + PRIOR_WEIGHT * prior
        evidence.append(Evidence(candidate=candidate, relevance=relevance,
                                  prior=prior, score=score))

    evidence.sort(key=lambda e: e.score, reverse=True)
    return [e for e in evidence if e.score >= MIN_SCORE]


def _relevance(query: str, candidates: list[Candidate]) -> list[float]:
    model = reranker()
    if model is not None:
        passages = [f"{c.row['title']} {c.row['section_path']}\n{c.row['body']}"
                    for c in candidates]
        return model.score(query, passages)

    top = max((c.rrf for c in candidates), default=0.0) or 1.0  # 模型不可用，退回融合分
    return [c.rrf / top for c in candidates]
```

## 九、装配：去重、相邻合并、父块回填、预算截断

子块负责检索精度，父块负责给模型完整规则。装配按 `parent_id` 去重，同文档内相邻父块（`parent_seq` 相差 1）合并成一组，最多合并 3 个；再按同一 `parent_id` 下全部子块拼接回填父块原文；最后按 token 预算截断，超预算先收缩到命中窗口再试一次。

```python
TOKEN_BUDGET = 3000
MERGE_MAX_PARENTS = 3

def assemble(evidence: list[Evidence], top_k: int) -> list[PolicySection]:
    groups = _group(evidence, top_k)
    sections, used = [], 0
    count = embedder().count_tokens

    for group in groups:
        text = _render(group)
        cost = count(text)
        if used + cost > TOKEN_BUDGET:
            text = _render(group, window=True)     # 收缩到命中子块前后各一块
            cost = count(text)
            if used + cost > TOKEN_BUDGET:
                break
        sections.append(_to_section(group, text))
        used += cost
    return sections


def _group(evidence: list[Evidence], top_k: int) -> list["_Group"]:
    groups, seen_parents = [], set()
    for item in evidence:
        parent = item.row["parent_id"]
        if parent in seen_parents:
            _parent_of(groups, parent).absorb(item)
            continue
        merged = next((g for g in groups if g.adjacent_to(item)), None)
        if merged is not None:
            merged.absorb(item)
        elif len(groups) < top_k:
            groups.append(_Group(item))
        else:
            continue
        seen_parents.add(parent)
    return groups


def _render(group, window: bool = False) -> str:
    parts = []
    for _, parent_id, _ in group.parents:      # 按 parent_seq 有序
        rows = store.fetch_parent(parent_id)
        if window:
            rows = _hit_window(rows, group.hit_chunks)
        parts.extend(r["body"] for r in rows)
    return "\n\n".join(parts)
```

## 十、异常与降级

| 组件 | 不可用时的行为 |
|---|---|
| 改写模型 | 原文透传，排序掉一档 |
| 重排模型 | 退回融合分 + 层级先验加权 |
| 嵌入模型 | 直接抛错，不切换向量空间 |
| 召回一条候选都没有 | 抛 `NoCandidatesError` |
| 重排 / 装配后无证据 | 抛 `NoEvidenceError` |

检索到空结果时不返回空列表交给下游继续判定——那等同于让上游凭记忆编造政策，所以两类失败都显式抛异常，而不是静默退化成"未检索到条款"这句话。

```python
class RetrievalError(RuntimeError):
    def __init__(self, message: str, trace=None) -> None:
        super().__init__(message)
        self.trace = trace          # 带上这次检索的中间产物，排查召回还是重排的问题

class NoCandidatesError(RetrievalError):
    """召回层没有任何候选，通常是知识库或过滤配置故障。"""

class NoEvidenceError(RetrievalError):
    """候选存在，但重排 / 装配后没有可交付证据。"""
```
