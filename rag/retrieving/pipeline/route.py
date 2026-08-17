"""Step 2 · 路由 —— 决定每条子查询打哪些层、各取多少。

这个知识库只有一个数据源（Milvus），但**内部分两层**，两层的角色完全不同
（doc/policy/README.md 第二节）：

- `platform` 是平台与消费者之间的**直接约定**，答复消费者时引用的就是它；
- `law` 是**法定底线**，用来判断平台条款是否有效、消费者是否另有救济途径。

所以路由要回答的是：这次检索该往哪一层倾斜。**它不是硬开关**——
两层始终都召回，路由只调节各自的名额与后续的加权（见 rerank.py）。
硬砍掉 law 会让「平台说不能退，但法律说可以」这类问题永远拿不到答案；
而让 law 和 platform 平权竞争同样有害：L02 说七日无理由，P02 说金牌会员 15 天，
两条同时进上下文、又没有效力位阶提示，模型很容易引错那条更严的。

举个具体的：问「签收 10 天还能退吗」，法规层 L02 的「七日无理由退货」措辞
和问题高度相似，向量分不低，但它不是答案 —— 答案是 P02 第二条的金牌 15 天。
默认配额（platform 18 / law 5）就是在这种情况下保证平台层不被挤出候选集。
"""

from dataclasses import dataclass

from rag.retrieving.pipeline.rewrite import RetrievalPlan, SubQuery

# 每条子查询、每一路（dense / BM25）的召回条数
PLATFORM_K = 18
LAW_K = 5

# 需要判断条款效力时，法规层与平台层平权
VALIDITY_PLATFORM_K = 12
VALIDITY_LAW_K = 12

LAW_INTENTS = {"validity", "dispute"}
"""这两类问题必须看法条：
- validity：用户在质疑平台规则本身，答案不在平台层；
- dispute：争议升级会走到投诉、仲裁、诉讼，法定途径是答复的一部分。"""


@dataclass
class Route:
    sub_query: SubQuery
    layer_k: dict[str, int]
    """每层各召回多少条。key 是 layer，同时也是过滤表达式的取值范围。"""
    law_weight: float
    """法规层在重排时的层级权重（rerank.py 用）。平台层恒为 1.0。"""

    @property
    def layers(self) -> list[str]:
        return list(self.layer_k)


def route(plan: RetrievalPlan) -> list[Route]:
    routes: list[Route] = []
    for sub in plan.sub_queries:
        # needs_law 是整个请求级的判断（用户在质疑规则合法性），
        # intent 是这条子查询自己的性质 —— 任一命中就提权法规层
        law_first = plan.needs_law or sub.intent in LAW_INTENTS
        if law_first:
            routes.append(
                Route(
                    sub_query=sub,
                    layer_k={"platform": VALIDITY_PLATFORM_K, "law": VALIDITY_LAW_K},
                    law_weight=1.0,
                )
            )
        else:
            routes.append(
                Route(
                    sub_query=sub,
                    layer_k={"platform": PLATFORM_K, "law": LAW_K},
                    # 法规层仍然召回、仍然可能进上下文，但排序上让位于平台条款。
                    # 0.5 不是调出来的值，是个保守起点 —— 要靠标注集校准。
                    law_weight=0.5,
                )
            )
    return routes
