"""rag-ex-1 的打分器。

这里只放**不调模型**的部分：三档 Recall 加两个辅助数。它们是纯函数，同一份数据集
跑两遍结果一致，因此能进门禁（5-rag-eval 六）。要调 judge 的 Context Recall 与
Context Relevance 在 `judge.py`，那两个有噪声，只进报告不进门禁。

三档取自链路的三个位置，读的不是同一个序列：

| 指标 | 读什么 | 诊断 |
|---|---|---|
| `recall@10` | 召回融合后的候选（`CANDIDATE_LIMIT = 20`） | 召回层的上界，这里没有的后面救不回来 |
| `recall@3` | 重排后前 3（`DEFAULT_TOP_K = 4`，留一格余量） | 实际交付水位 |
| `recall@1` | 重排后第 1 | 头部精度，对 `MIN_SCORE` 最敏感 |
"""

from rag.retrieving.protocol import PolicySection


def recall_at_k(retrieved: list[str], seeds: list[str], k: int, equivalent: list[list[str]] | None = None) -> float:
    """全部种子块都落在前 k 个里才算命中，不给部分分。

    `multi_hop` 的两个种子块是一套规则的两半，只召回一半答案照样是错的
    （4-rag-dataset 5.4）。给部分分会把「答不全」和「答对了」在均值里混成一个
    中间数，分档报表也就跟着失去意义。
    """
    groups = equivalent or [[seed] for seed in seeds]
    return float(all(any(chunk_id in retrieved[:k] for chunk_id in group) for group in groups))


def seed_ranks(retrieved: list[str], seeds: list[str]) -> dict[str, int | None]:
    """种子块在这个序列里的名次（1 起），没出现记 null。

    Recall 是 0/1，掉分时看不出差多少。名次能：候选里排第 12 说明召回够了、是
    截断或重排的事；候选里根本没有，就得回去看切片、块头和过滤条件。
    """
    return {seed: (retrieved.index(seed) + 1 if seed in retrieved else None) for seed in seeds}


def evidence_tokens(sections: list[PolicySection]) -> int:
    """证据的实际 token 用量。

    `TOKEN_BUDGET = 3000` 是上限不是配额，凑满没有好处。它和 Recall 一起看才知道
    一次调参是净赚还是净亏：分数涨了但 token 翻倍，涨的是塞进去的量不是检索质量。
    """
    from llm.embedding import embedder

    count = embedder().count_tokens
    return sum(count(s.text) for s in sections)


def duplicate_ratio(sections: list[PolicySection]) -> float:
    """同一批装配上下文中，重复段落占总字符数的比例。

    这是一个内容重复信号，不证明重复一定由装配造成；不同父块或不同文档本身
    也可能包含相同的标准表述。具体归因要结合 trace 和装配测试。
    """
    seen: set[str] = set()
    duplicated = total = 0
    for section in sections:
        for para in section.text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            total += len(para)
            if para in seen:
                duplicated += len(para)
            else:
                seen.add(para)
    return round(duplicated / total, 3) if total else 0.0
