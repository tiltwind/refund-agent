"""Step 1 · 改写 —— 把一句口语变成可执行的检索意图。

Agent 传进来的 query 已经是半结构化的（prompt 要求它把「类目 + 签收天数 +
会员等级 + 诉求」拼进同一个 query），所以这一步**不做关键词提取**，只做三件
模型真正擅长的事：

1. **拆多意图**：「能退吗？运费谁出？」是两个检索目标，命中的是 P02 和 P06
   两篇不同的文档。合成一条 query 会让两边都召回不全 —— 向量会落在两个主题
   的语义质心上，哪篇都不像。
2. **判断要不要法规层**：只有「平台这么规定合法吗」「我还有别的救济途径吗」
   这类问题才需要召回 law（见 route.py）。
3. **还原成问句**（见下）。

## 为什么必须是问句

这一步最反直觉的地方：改写的输出**不是关键词串，是完整的自然语言问句**。

Agent 按 prompt 拼出来的 query 长这样：`金牌会员 耳机 未拆封 签收10天 无理由退货`。
实测下来，同一个语料、同一套参数，只把它改写成
`金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？`，
重排的 top-1 就从 P07 第三条（极速退款，与问题无关，只是「金牌会员」四个字
高度匹配）变成了 P02 第二条（退货窗口，正确答案），分数 0.947 → 0.984。

原因在重排那一步：cross-encoder 判断的是「这段文字**回没回答这个问题**」，
而关键词串里根本没有问题 —— 没有疑问点，它只能退化成算主题相似度，于是
「一堆金牌会员相关的条款」就都长得一样了。稠密检索同理：BGE-M3 的训练语料
里 query 侧是自然提问，喂关键词串属于分布外输入。只有 BM25 不在乎语序。

所以「精简成关键词」这个看起来天经地义的检索预处理，在带重排的链路里是**反
向优化**。要删的是寒暄，不是句子结构。

**模型不许碰的东西**：生效日期由代码用 `date.today()` 算（filters.py），
不进 prompt 也不由模型输出。模型对「今天几号」没有可靠认知，把它交出去等于
让检索随机漏掉刚生效或刚废止的条款。

改写有 100~300ms 延迟和被改坏的风险，所以**失败一律降级为原文透传**，
绝不因为改写挂掉整条检索链路。代价要认清楚：透传时进重排的就是那个关键词串，
排序质量按上面的实测会掉一档 —— 正确条款仍在 top-4 里（模型看得到），
但不再稳定排第一。没有 ANTHROPIC_API_KEY 时走的就是这条路径。
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

MODEL = os.getenv("REFUND_AGENT_REWRITE_MODEL", "anthropic:claude-haiku-4-5")
"""改写是分类式的轻任务 —— 换更便宜、更快的档次比用主模型划算得多。"""

ENABLED = os.getenv("REFUND_AGENT_REWRITE", "on").lower() not in ("off", "0", "false")

TIMEOUT_SECONDS = float(os.getenv("REFUND_AGENT_REWRITE_TIMEOUT", "3"))

Intent = Literal[
    "window",     # 退货窗口、时限        → P02 第二条 / P07
    "condition",  # 商品状态、完好标准     → P02 第三条 / P05
    "category",   # 类目限制             → P02 第四条 / P04 / P11
    "risk",       # 风控、高风险账户       → P02 第五条 / P08
    "refund",     # 退款金额、到账、运费、券 → P03 / P06 / P10
    "dispute",    # 争议、平台介入、维权    → P09
    "validity",   # 条款效力、法定标准      → 法规层
    "other",
]


# ── 模型输出的结构 ────────────────────────────────────────────────────────
class SubQuery(BaseModel):
    id: str
    intent: Intent
    text: str = Field(
        description="一个完整的自然语言问句，把该检索目标问清楚，"
        "并带上类目、签收天数、会员等级、商品状态等已知条件"
    )


class Rewrite(BaseModel):
    sub_queries: list[SubQuery] = Field(description="1~3 条，按检索目标拆分")
    needs_law: bool = Field(
        description="是否需要召回法律法规层：用户质疑平台规则是否合法、"
        "主张法定权利、或询问平台之外的救济途径时为 true"
    )


# ── 本地结构 ──────────────────────────────────────────────────────────────
@dataclass
class RetrievalPlan:
    original: str
    sub_queries: list[SubQuery]
    needs_law: bool
    rewritten: bool
    """False 表示走了降级路径（未启用 / 无 API key / 调用失败），原文透传。"""


SYSTEM = """你是退款政策检索链路的查询改写器。把客服 Agent 的检索请求转成结构化检索意图。

1. 只输出改写结果，不回答问题、不臆造事实、不补充政策内容。
2. 一个请求包含多个检索目标时拆成多条 sub_query（最多 3 条）；只有一个目标就只出一条。
   例：「耳机拆封了能退吗，运费谁出」→ condition 一条 + refund 一条。
3. sub_query.text 必须是一个**完整的自然语言问句**，不是关键词串。
   把已知条件（类目、签收天数、会员等级、商品状态、退款次数）写进句子，去掉寒暄。
   ✓「金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？」
   ✗「金牌会员 耳机 未拆封 签收10天 无理由退货」
   这一条是硬要求，理由见本文件模块 docstring 的「为什么必须是问句」。
4. needs_law 只在用户**质疑平台规则本身**（「你们这规定合法吗」）、主张法定权利
   （「消法规定七天无理由」）或问平台外救济途径（「我要投诉/起诉」）时为 true。
   普通的「我能不能退」一律 false —— 答复消费者引用的是平台条款。
5. 不要输出任何日期。生效日期由系统按当天计算，你无从判断今天是几号。"""


@lru_cache(maxsize=1)
def _model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        MODEL, temperature=0, timeout=TIMEOUT_SECONDS, max_retries=1
    ).with_structured_output(Rewrite)


def _passthrough(query: str) -> RetrievalPlan:
    return RetrievalPlan(
        original=query,
        sub_queries=[SubQuery(id="q1", intent="other", text=query)],
        needs_law=False,
        rewritten=False,
    )


def rewrite(query: str) -> RetrievalPlan:
    if not ENABLED:
        return _passthrough(query)
    try:
        result = _model().invoke(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        )
    except Exception as exc:  # noqa: BLE001
        # 改写不是必选项：挂了就原文透传，检索照常进行
        print(f"[warn] 查询改写失败（{type(exc).__name__}: {exc}），降级为原文透传")
        return _passthrough(query)

    if not result.sub_queries:
        return _passthrough(query)
    return RetrievalPlan(
        original=query,
        sub_queries=result.sub_queries[:3],
        needs_law=result.needs_law,
        rewritten=True,
    )
