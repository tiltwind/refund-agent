"""语义切分 —— 只用在「一个自然段本身就超长」的兜底位置。

**为什么不把它当主策略**：多项独立评测（Chroma 的 token 级报告、学术界的
对比研究）给出一致结论 —— 语义切分相对结构/递归切分的提升通常只有 1~5 个
百分点，计算成本却高一个数量级（要对全部句子做 embedding）。它真正有价值的
场景是「文档没有任何显式结构且话题切换频繁」，而 doc/policy 下的文档恰恰相反：
标题层级规范，一个 `##` 就是一条完整规则。

**那什么时候用**：段落切分之后仍有个别自然段超过子块硬上限。这时候按标题切
已经无路可走，按字符硬切又会从句子中间劈开，语义切分是这里唯一讲道理的选择。
实际语料上命中这条路径的段落是个位数 —— 成本可以忽略，收益是那几段不被切碎。
"""

import re
from collections.abc import Callable

_SENT_END = re.compile(r"(?<=[。！？；!?;])\s*")

BREAK_PERCENTILE = 85
"""断点分位数。相邻句距离排进前 15% 才算「话题变了」。

调高 → 断点更少、块更大；调低 → 切得更碎。85 是个保守值：宁可块偏大，
也不要在法条的「本条所称……」和它的定义之间切一刀。
"""

MIN_FILL = 0.5
"""已积累到目标长度的这个比例之前，不允许因为语义断点收尾 ——
否则一段「总则一句 + 细则五句」的文字会在第一句后就被切开，
留下一个 20 token、没有任何检索价值的碎块。"""


def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句，标点跟随前句。换行也算边界（列表项各成一句）。"""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.extend(s for s in (p.strip() for p in _SENT_END.split(line)) if s)
    return out


def semantic_split(
    text: str,
    encode: Callable[[list[str]], list[list[float]]],
    tokens: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    """把超长段落切成若干块，尽量落在话题切换处。"""
    sents = split_sentences(text)
    # 句子太少，语义断点无从谈起（分位数在 2 个样本上没有意义），直接按长度硬切
    if len(sents) < 3:
        return _hard_split(sents or [text], tokens, max_tokens)

    vecs = encode(sents)  # BGE-M3 输出已 L2 归一化，点积即余弦
    dists = [1.0 - sum(a * b for a, b in zip(vecs[i], vecs[i + 1])) for i in range(len(sents) - 1)]
    threshold = _percentile(dists, BREAK_PERCENTILE)

    chunks: list[str] = []
    cur: list[str] = [sents[0]]
    cur_tokens = tokens(sents[0])

    for i, sent in enumerate(sents[1:]):
        n = tokens(sent)
        over_budget = cur_tokens + n > max_tokens
        topic_shift = dists[i] >= threshold and cur_tokens >= target_tokens * MIN_FILL
        if cur and (over_budget or topic_shift):
            chunks.append("".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sent)
        cur_tokens += n

    if cur:
        chunks.append("".join(cur))

    # 单句就超上限的极端情况，最后再按 token 硬切一次兜底
    out: list[str] = []
    for c in chunks:
        out.extend(_hard_split([c], tokens, max_tokens) if tokens(c) > max_tokens else [c])
    return out


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[idx]


def _hard_split(parts: list[str], tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    """最后的兜底：按字符二分逼近 token 上限。

    到这一步语义已经保不住了，唯一的目标是别让内容被**静默截断** ——
    超出 max_seq_length 的部分对向量的贡献严格为 0，那段知识看起来在库里，
    实际上永远召回不到。宁可切丑，不能丢。
    """
    out: list[str] = []
    for part in parts:
        while tokens(part) > max_tokens:
            lo, hi = 1, len(part)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if tokens(part[:mid]) <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            out.append(part[:lo])
            part = part[lo:]
        if part:
            out.append(part)
    return out
