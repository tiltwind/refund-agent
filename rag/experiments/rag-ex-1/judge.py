"""rag-ex-1 的两个 LLM 指标：Context Recall 与 Context Relevance（5-rag-eval 七）。

    Context Recall    = 被检回上下文支撑的 claim 数 / 参考答案中的 claim 总数
    Context Relevance = 与 query 相关的内容单元数 / 检回的内容单元总数

跟 `scorers.py` 分开放，因为这两个不是纯函数：判分要调 judge 模型，同一份数据跑
两遍可能给出不同的数。三档 Recall 进门禁，这两个只进报告（README 五）。

判分挂在 `PolicySection.text` 上 —— 那是真正注入模型上下文的东西。`PolicySection`
上没有 `chunk_id`（装配会把子块还原成完整小节、还会合并相邻父块），所以这两个指标
读的是正文，不是 ID。

## 两个指标的方向相反

Context Recall 问「捞得够不够全」，Context Relevance 问「捞回来的有多少是废的」。
只报前者，把 `top_k` 从 4 改成 20 分数就变好看 —— 而 `TOKEN_BUDGET = 3000` 是硬
上限。两个一起看才知道一次调参是净赚还是净亏。

Context Relevance 的绝对值不该追求高：内容单元取的是句子，而一条 `PolicySection`
是回填后的完整小节，里面必然混着不相关的句子 —— 那是父块回填故意带进来的
（3-rag-impl 五）。按块判会把这个设计判成缺陷。它的用途是横向对比。

## judge 的四条约束

写进提示词、也体现在这里的代码里：温度 0 且结构化输出（`common.build_judge`）；
不确定一律判负；只判「上下文里有没有」，不判「答案对不对」；judge 模型与链路的
改写模型分开配（撞车时 `judge_name()` 会打警告）。

判定理由逐条落盘到 `result.json` 的 `judge` 字段 —— 一个 0.6 分说明不了改哪里，
要能翻到是哪条 claim 没被支撑。

接进来之前 judge 本身要先校准（4-rag-dataset 8.2 的人工抽检样本），校准之前这两个
分数只作观察。
"""

import re
from functools import lru_cache

from pydantic import BaseModel, Field

from rag.evals.common import build_judge, judge_json_hint
from rag.retrieving.protocol import PolicySection

MAX_UNITS = 300
"""一次判定最多送多少个内容单元。`TOKEN_BUDGET = 3000` 之下正常在 40~80 之间，
超过这个数说明装配那边出了事，截断比让 judge 吃一个畸形输入更好定位。"""

TIMEOUT_SECONDS = 120.0
"""judge 开着思考模式判一百多个内容单元，实测 60s 会偶尔超时（超时那条不写分数，
均值的分母就少一个）。宁可等，别丢样本。"""

_SENTENCE_END = re.compile(r"(?<=[。；！？])")


# ── 模型输出的结构 ────────────────────────────────────────────────────────
class ClaimVerdict(BaseModel):
    index: int = Field(description="claim 的编号，从 1 开始")
    supported: bool = Field(description="上下文里能不能找到支撑这条 claim 的内容")
    reason: str = Field(description="30 字以内：支撑它的是哪一段，或者缺了什么")


class RecallVerdict(BaseModel):
    verdicts: list[ClaimVerdict] = Field(description="逐条判定，每条 claim 一项，不要漏")


class RelevanceVerdict(BaseModel):
    relevant: list[int] = Field(description="对回答这个问题有用的内容单元编号，从 1 开始")
    note: str = Field(description="30 字以内：不相关的那些主要是什么")


RECALL_SYSTEM = """你在评测一个检索系统。给你一段检回的上下文和若干条 claim，逐条判断
上下文能不能支撑这条 claim。

1. 只判「上下文里有没有这个信息」，**不判这条 claim 本身对不对**。claim 说错了但上下文里
   确实这么写，判 supported=true。
2. 表述不同、意思相同算支撑；claim 的信息散在上下文的几段里、拼起来能得到，也算支撑。
3. 上下文只沾到主题、没有给出 claim 说的那个具体条件/期限/数字，判 supported=false。
4. **不确定一律判 false。** 靠推理、常识或你自己知道的政策补出来的，都不算支撑。
5. 每条 claim 出一项判定，编号与输入一致，不要合并、不要漏。"""

RELEVANCE_SYSTEM = """你在评测一个检索系统。给你一个用户问题和检回上下文切出的若干个内容单元，
挑出对回答这个问题**有用**的那些。

1. 有用 = 这个单元本身构成回答该问题的依据（条件、期限、金额、判定标准、例外、责任归属）。
2. 同主题但答的是另一种情形的条款，不算有用。举例：问「拆封了能不能退」，
   「运费由谁承担」不算有用。
3. 小节标题、表格的表头行：只有当它给出了回答该问题所需的信息时才算有用。
4. **不确定一律不选。**
5. 只输出编号，不要改写内容。"""


# ── 内容单元 ──────────────────────────────────────────────────────────────
def units(sections: list[PolicySection]) -> list[str]:
    """把检回的上下文切成判定单元：句子一条，表格行整行一条。

    取句子而不是取 `PolicySection`：一条 section 是回填后的完整小节，按块判的话
    「小节里混着不相关的句子」这个事实根本表达不出来，分数会一律接近 1。
    表格行不按句号切 —— `| 已拆封 | 不支持 | …` 里没有句号，切了也是把一行拆成
    几个没有主语的碎片。
    """
    out: list[str] = []
    for section in sections:
        for line in section.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("|"):
                out.append(line)
                continue
            out += [part.strip() for part in _SENTENCE_END.split(line) if part.strip()]
    return out


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))


def _fail(reason: str) -> dict:
    """judge 调用失败。这条要报出来 —— 一次网关抖动会静悄悄地把指标算在少一半样本上。"""
    return {"score": None, "error": reason}


def _skip(reason: str) -> dict:
    """没有可判的东西，不是故障：分母是 0，这条本来就没有这个指标。

    与 `_fail` 分开，否则空证据（6.1 那批）每跑一次都会在末尾报一串「judge 失败」，
    真正的网关故障淹在里面看不出来。
    """
    return {"score": None, "error": None, "skipped": reason}


# ── 两个指标 ──────────────────────────────────────────────────────────────
def context_recall(claims: list[str], sections: list[PolicySection]) -> dict:
    """检回的上下文撑不撑得住参考答案。claims 由 `rag/evals/generate_claims.py` 预先拆好。"""
    if not claims:
        return _skip("样本没有 claim")
    if not sections:
        # 空证据不必花钱问模型：一条都撑不住。这是 6.1 那个坏 case 的常态
        return {"score": 0.0, "n": len(claims), "hit": 0, "error": None,
                "detail": [{"claim": c, "supported": False, "reason": "上下文为空"} for c in claims]}

    context = "\n\n".join(f"【{s.section}】\n{s.text}" for s in sections)
    user = f"### 检回的上下文\n{context}\n\n### claim\n{_numbered(claims)}"
    try:
        result = _model(RecallVerdict).invoke(
            [{"role": "system", "content": RECALL_SYSTEM + judge_json_hint(RecallVerdict)},
             {"role": "user", "content": user}]
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")

    # 模型漏判的 claim 按不支撑计 —— 分母是数据集定的 claim 总数，不是模型返回的条数，
    # 否则「少判几条」会把分数抬上去
    by_index = {v.index: v for v in result.verdicts}
    detail = []
    for i, claim in enumerate(claims, 1):
        verdict = by_index.get(i)
        detail.append({
            "claim": claim,
            "supported": bool(verdict and verdict.supported),
            "reason": verdict.reason if verdict else "judge 漏判，按不支撑计",
        })
    hit = sum(1 for d in detail if d["supported"])
    return {"score": round(hit / len(claims), 3), "n": len(claims), "hit": hit,
            "detail": detail, "error": None}


def context_relevance(query: str, sections: list[PolicySection]) -> dict:
    """检回的东西里有多少是真跟问题相关的。不需要任何标注字段。"""
    items = units(sections)
    if not items:
        # 重排把候选全滤光时会走到这里（6.1）。相关度没有分母，与「判出来是 0」不同
        return _skip("上下文为空")
    truncated = len(items) > MAX_UNITS
    items = items[:MAX_UNITS]

    user = f"### 用户问题\n{query}\n\n### 内容单元\n{_numbered(items)}"
    try:
        result = _model(RelevanceVerdict).invoke(
            [{"role": "system", "content": RELEVANCE_SYSTEM + judge_json_hint(RelevanceVerdict)},
             {"role": "user", "content": user}]
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")

    # 越界编号丢掉：模型偶尔会返回一个不存在的序号，照单全收会算出大于 1 的分数
    relevant = {i for i in result.relevant if 1 <= i <= len(items)}
    return {
        "score": round(len(relevant) / len(items), 3),
        "n": len(items),
        "hit": len(relevant),
        "truncated": truncated,
        "note": result.note,
        "irrelevant": [items[i - 1][:60] for i in range(1, len(items) + 1) if i not in relevant][:10],
        "error": None,
    }


@lru_cache(maxsize=2)
def _model(schema):
    """两个指标各一个模型实例，按 schema 缓存。"""
    return build_judge(schema, timeout=TIMEOUT_SECONDS, max_retries=2)
