"""Step 4 · 召回融合 —— 稠密 + BM25 双路，RRF 合并。

## 为什么必须是两路

两路各自能捞到对方完全漏掉的东西，在这个语料上尤其明显：

- **BM25 不可替代**的是精确 term：`7 天`、`15 天`、`3 次`、`90 天`、`P02`、
  `生鲜`、`运费险`。用户问「近 90 天退款几次算高风险」，`90` 和 `3` 是低频
  高 IDF term，BM25 直接把 P02 第五条顶到第一；稠密检索只会把它稀释成
  「风控相关」这样的模糊语义，和 P08 整篇都很像。
- **稠密不可替代**的是措辞不搭的语义命中：用户说「拆开看了一眼就想退」，
  正文写的是「已开启包装，但未投入使用」，一个查询词都不重合。

## 为什么在应用层做 RRF 而不是用 Milvus 的 hybrid_search

Milvus 自带 `hybrid_search` + `RRFRanker`，一次调用就能出融合结果。这里不用它，
是因为融合分是**排查坏 case 的关键中间产物**：一条条款到底是两路都召回了、
还是只被 BM25 单路捞到（那它在 RRF 里天然吃亏，得靠重排救回来），
黑盒融合看不出来。多几次 RPC 换全链路可观测，这个交换在几百个块的规模上稳赚。

## 为什么用 RRF 而不是加权求和

BM25 分数无上界，cosine 在 [-1, 1]，两个尺度根本不可比。RRF 只用**排名**，
天然绕开归一化这件事 —— 这也是 Elasticsearch、Qdrant 把它做成内置融合选项的
原因。代价是丢掉了分数的绝对信息（第 1 名比第 2 名强多少），所以它只负责
粗筛出候选集，精排交给下一步。

`k=20` 而不是论文里的 60：60 是 TREC 规模语料的经验值，小语料上让头部排名的
权重更高更合适。这个值同样需要标注集校准，不是拍脑袋的定论。
"""

from dataclasses import dataclass, field

from llm.embedding import embedder
from rag.retrieving import store
from rag.retrieving.pipeline.filters import build_filter
from rag.retrieving.pipeline.route import Route

RRF_K = 20
CANDIDATE_LIMIT = 20
"""送进重排的候选数。重排是 cross-encoder，成本随候选数线性上升，
而 20 之后基本捞不到新东西了（几百个块的语料，相关的就那么几条）。"""


@dataclass
class Candidate:
    row: dict
    rrf: float = 0.0
    hits: list[str] = field(default_factory=list)
    """命中来源，形如 `q1/platform/bm25#2` —— 排查时一眼看出是谁把它捞出来的。"""

    @property
    def chunk_id(self) -> str:
        return self.row["chunk_id"]

    @property
    def single_path(self) -> bool:
        """只被一路召回。RRF 对这类候选天然不利（只有一个 1/(k+rank) 项），
        但它们往往正是稠密或 BM25 各自的独门收获 —— 别在这一步淘汰它们。"""
        return len(self.hits) == 1


def recall(routes: list[Route]) -> list[Candidate]:
    pool: dict[str, Candidate] = {}
    model = embedder()

    for route_ in routes:
        query = route_.sub_query.text
        vector = model.encode_query(query)
        for layer, k in route_.layer_k.items():
            # 按层分别检索，才能给每层单独的名额 —— 一次查两层再截断，
            # 平台条款会被措辞更像的法条挤出去（见 route.py）
            expr = build_filter([layer])
            tag = f"{route_.sub_query.id}/{layer}"
            _fuse(pool, store.search_dense(vector, expr, k), f"{tag}/dense")
            _fuse(pool, store.search_bm25(query, expr, k), f"{tag}/bm25")

    ranked = sorted(pool.values(), key=lambda c: c.rrf, reverse=True)
    return ranked[:CANDIDATE_LIMIT]


def _fuse(pool: dict[str, Candidate], rows: list[dict], tag: str) -> None:
    """把一路的排名列表并进候选池：score(d) += 1 / (k + rank(d))。"""
    for rank, row in enumerate(rows, start=1):
        candidate = pool.get(row["chunk_id"])
        if candidate is None:
            candidate = pool[row["chunk_id"]] = Candidate(row=row)
        candidate.rrf += 1.0 / (RRF_K + rank)
        candidate.hits.append(f"{tag}#{rank}")
