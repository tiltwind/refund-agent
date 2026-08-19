"""两个检索指标，都由 LLM judge 判定。

    Context Precision = Σ(Precision@k × rel_k) / 相关条数
    Context Recall    = 被上下文支撑的句子数 / ground_truth 的句子数

判定对象是 `search_policy` 返回的 `PolicySection` 列表 —— 那是真正注入模型的
上下文，按重排分降序，Context Precision 的位置权重就落在这个次序上。

judge 的四条约束写进提示词、也体现在代码里：温度 0 且结构化输出
（`common.build_judge`）；不确定一律判负；只判「上下文里有没有」，不判「答案
对不对」；judge 模型与链路的改写模型分开配（撞车时 `judge_name()` 打警告）。

逐条判定理由落进 `result.json` —— 一个 0.6 分说明不了改哪里。
"""

import re
from functools import lru_cache

from pydantic import BaseModel, Field

from rag.evals.common import build_judge, judge_json_hint
from rag.retrieving.protocol import PolicySection

TIMEOUT_SECONDS = 120.0
"""judge 开着思考模式判一段长上下文，实测 60s 会偶尔超时。超时那条不写分数，
均值的分母就少一个 —— 宁可等。"""

_SENTENCE_END = re.compile(r"(?<=[。！？；])")


# ── 模型输出的结构 ────────────────────────────────────────────────────────
class Verdict(BaseModel):
    index: int = Field(description="编号，从 1 开始，与输入一致")
    hit: bool = Field(description="判定结果")
    reason: str = Field(description="30 字以内的理由")


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(description="逐条判定，每条一项，不要合并、不要漏")


PRECISION_SYSTEM = """你在评测一个检索系统。给你一个用户问题和检回的若干段政策条款，
逐段判断这一段对回答该问题有没有用。

1. 有用 = 这一段本身给出了回答该问题所需的依据（条件、期限、金额、判定标准、例外、责任归属）。
2. 同一主题但答的是另一种情形的条款，判 hit=false。举例：问「拆封了能不能退」，
   「运费由谁承担」判 false。
3. 一段里只要有一部分构成依据就判 true —— 条款是按小节回填的，夹带相邻正文属于正常。
4. **不确定一律判 false。**
5. 每段出一项判定，编号与输入一致。"""

RECALL_SYSTEM = """你在评测一个检索系统。给你一段检回的上下文和标准答案拆出的若干个句子，
逐句判断上下文能不能支撑这句话。

1. 只判「上下文里有没有这个信息」，**不判这句话本身对不对**。句子说错了但上下文里
   确实这么写，判 hit=true。
2. 表述不同、意思相同算支撑；信息散在上下文的几段里、拼起来能得到，也算支撑。
3. 上下文只沾到主题、没有给出这句话说的那个具体条件/期限/数字，判 hit=false。
4. **不确定一律判 false。** 靠推理、常识或你自己知道的政策补出来的，都不算支撑。
5. 每句出一项判定，编号与输入一致。"""


# ── 两个指标 ──────────────────────────────────────────────────────────────
def context_precision(question: str, sections: list[PolicySection]) -> dict:
    """检回的条款里有多少是真有用的，且有用的排得够不够靠前。

    位置加权取自 RAGAS：第 k 段判为相关时，它贡献的是前 k 段的精确率
    `Precision@k`，因此同样两段相关，排在 1、2 位得 1.0，排在 3、4 位只得 0.58。
    检索交付给模型的是一个有序列表，把不相关的顶在前面本身就是缺陷。

    这个指标不需要任何标注字段 —— 只看问题和检回的内容。
    """
    if not sections:
        # 一条都没检回时分母是 0，数学上未定义，判分口径取 0：这是链路的失败结果，
        # 不是「没有可判的东西」。跳过它会把最差的那批样本从精度均值里摘出去，
        # 两个指标的分母也就对不上，并排读不出东西
        return {"score": 0.0, "n": 0, "hit": 0, "error": None,
                "detail": [{"text": "（无）", "hit": False, "reason": "没有检回任何条款"}]}

    items = [f"【{s.section}】\n{s.text}" for s in sections]
    hits = _judge(PRECISION_SYSTEM, f"### 用户问题\n{question}\n\n### 检回的条款\n{_numbered(items)}", items)
    if "error" in hits:
        return hits

    flags = [v["hit"] for v in hits["detail"]]
    total = sum(flags)
    if not total:
        return {"score": 0.0, "n": len(flags), "hit": 0, "detail": hits["detail"], "error": None}
    # Σ(Precision@k × rel_k) / 相关条数：只在相关的位置上累加，分母是相关条数不是总条数
    weighted = sum(
        sum(flags[: k + 1]) / (k + 1)
        for k in range(len(flags))
        if flags[k]
    )
    return {"score": round(weighted / total, 3), "n": len(flags), "hit": total,
            "detail": hits["detail"], "error": None}


def context_recall(ground_truth: str, sections: list[PolicySection]) -> dict:
    """回答这个问题需要的信息，检回的上下文里有没有。

    标准答案按句拆开逐句归因，分母是句子数。拆句是纯函数（`sentences`），
    同一条 ground_truth 每次拆出的条数相同，两次 run 的分母才可比。
    """
    parts = sentences(ground_truth)
    if not parts:
        return _skip("样本没有 ground_truth")
    if not sections:
        # 空上下文不必花钱问模型：一句都撑不住
        return {"score": 0.0, "n": len(parts), "hit": 0, "error": None,
                "detail": [{"text": p, "hit": False, "reason": "上下文为空"} for p in parts]}

    context = "\n\n".join(f"【{s.section}】\n{s.text}" for s in sections)
    hits = _judge(RECALL_SYSTEM, f"### 检回的上下文\n{context}\n\n### 标准答案的句子\n{_numbered(parts)}", parts)
    if "error" in hits:
        return hits

    hit = sum(1 for v in hits["detail"] if v["hit"])
    return {"score": round(hit / len(parts), 3), "n": len(parts), "hit": hit,
            "detail": hits["detail"], "error": None}


def sentences(text: str) -> list[str]:
    """把标准答案切成判定单元：按句末标点与分号切。

    分号也切，中文政策答案里它分隔的是并列条款（「运费由消费者承担；可选择上门
    取件」），两半各自完整、各自可判。逗号不切 —— 切出来是缺主语的碎片。

    拆句是纯函数，同一条 ground_truth 每次拆出的条数相同。代价是分母粗：
    96 条样本平均 1.58 句，只有一句的那批 Context Recall 非 0 即 1，
    单条读不出程度，均值才有意义。"""
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


# ── 判定 ──────────────────────────────────────────────────────────────────
def _numbered(items: list[str]) -> str:
    return "\n\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))


def _judge(system: str, user: str, items: list[str]) -> dict:
    """调一次 judge，把返回的判定按编号对回原始条目。

    模型漏判的按判负计 —— 分母是输入的条数，「少判几条」不该把分数抬上去。
    """
    try:
        result = _model().invoke(
            [{"role": "system", "content": system + judge_json_hint(Verdicts)},
             {"role": "user", "content": user}]
        )
    except Exception as exc:  # noqa: BLE001
        # judge 调用失败不写分数：写 0 会被均值当成检索没召回，把模型故障记到检索头上
        return {"score": None, "error": f"{type(exc).__name__}: {exc}"}

    by_index = {v.index: v for v in result.verdicts}
    detail = []
    for i, text in enumerate(items, 1):
        verdict = by_index.get(i)
        detail.append({
            "text": text[:120],
            "hit": bool(verdict and verdict.hit),
            "reason": verdict.reason if verdict else "judge 漏判，按判负计",
        })
    return {"detail": detail}


def _skip(reason: str) -> dict:
    """没有可判的东西，不是故障：分母是 0，这条本来就没有这个指标。

    与判定失败分开 —— 失败清单只留真正的网关故障，才有报出来的价值。
    """
    return {"score": None, "error": None, "skipped": reason}


@lru_cache(maxsize=1)
def _model():
    """两个指标共用一个 judge 实例：判定结构相同，都是逐条 0/1 加一句理由。"""
    return build_judge(Verdicts, timeout=TIMEOUT_SECONDS, max_retries=2)
