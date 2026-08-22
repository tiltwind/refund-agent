"""Step 6 · 装配 —— 小块检索、大块喂模型。

四件事，顺序不能颠倒：**去重 → 相邻合并 → 父块回填 → 预算截断**。

## 为什么要回填父块

子块是为**检索精度**切的（目标 320 token，一段条文），它对「签收 10 天能退吗」
这类事实型问题刚刚好。但答复消费者时，只给「2.3 窗口判定为大于即超出」这一段
是不够的 —— 上一段的 7 天 / 15 天窗口值才是结论的依据。父块回填就是把命中的
子块还原成它所在的完整小节，让模型拿到成套的规则。

父块不单独存储：它就是同 parent_id 的子块按 chunk_index 拼接
（rag/chunking/model.py）。子块之间 overlap = 0，拼接精确还原原文。

## 为什么要在这里合并相邻父块

法规层的文档（L04、L05）是 `#` 直接跳到 `###`，父块退化成了单条条文
（rag/chunking/policy.py 里说明了为什么不在索引期补救）。当「第九条
售出 7 日内」和「第十条 售出 15 日内」同时命中时，它们本就是一套规则的两半，
分开注入会让模型看到两条孤立的期限。**合并的依据是「这次确实都命中了」**，
比索引期按大小硬凑父块可靠。

## 预算是上限不是配额

给证据的预算只有 3000 token，远小于模型窗口。三个理由：塞得越多越容易触发
中间遗忘（关键证据排在中部时模型经常用不上）；输入 token 直接计费；上下文
越长首 token 延迟越高。**凑满预算没有任何好处** —— 实际用量通常远低于它。
"""

import os

from llm.embedding import embedder
from rag.retrieving import store
from rag.retrieving.pipeline.rerank import Evidence
from rag.retrieving.protocol import PolicySection

TOKEN_BUDGET = int(os.getenv("REFUND_AGENT_POLICY_BUDGET", "3000"))


def assemble(evidence: list[Evidence], top_k: int) -> list[PolicySection]:
    groups = _group(evidence, top_k)

    sections: list[PolicySection] = []
    used = 0
    count = embedder().count_tokens

    for group in groups:
        text = _render(group)
        cost = count(text)
        if used + cost > TOKEN_BUDGET:
            # 超预算不是直接丢弃 —— 先收缩到命中窗口（命中子块及其前后各一块）
            # 再试一次。整条丢掉可能丢的正是排名第一的那条。
            text = _render(group, window=True)
            cost = count(text)
            if used + cost > TOKEN_BUDGET:
                break
        sections.append(_to_section(group, text))
        used += cost

    return sections


# ── 分组：按父块去重 + 相邻合并 ───────────────────────────────────────────
MERGE_MAX_PARENTS = 3
"""相邻合并的上限。合并是为了把被切碎的条文拼回一套规则，不是为了把半篇
文档灌进去 —— 无上限地滚雪球会让一次命中拖进十条不相关的邻条。"""


class _Group:
    """一块要注入上下文的证据，可能横跨若干相邻父块。"""

    def __init__(self, evidence: Evidence) -> None:
        self.best = evidence
        # (parent_seq, parent_id, section_path)，始终按文档内顺序保持有序 ——
        # 条款要按原文先后读才成立，渲染和命名都依赖这个顺序
        self.parents: list[tuple[int, str, str]] = [_key(evidence)]
        self.hit_chunks: set[str] = {evidence.row["chunk_id"]}

    @property
    def doc_id(self) -> str:
        return self.best.row["doc_id"]

    def absorb(self, evidence: Evidence) -> None:
        self.hit_chunks.add(evidence.row["chunk_id"])
        key = _key(evidence)
        # 判重只看 parent_id：同一父块下的子块 section_path 可能不同
        # （一条下面挂多个子标题），按完整 key 判重会把同一父块记两遍，
        # _render 就会把它的正文拼两遍。
        if not any(parent_id == key[1] for _, parent_id, _ in self.parents):
            self.parents.append(key)
            self.parents.sort()

    def adjacent_to(self, evidence: Evidence) -> bool:
        if evidence.row["doc_id"] != self.doc_id:
            return False
        if len(self.parents) >= MERGE_MAX_PARENTS:
            return False
        return any(abs(evidence.row["parent_seq"] - seq) == 1 for seq, _, _ in self.parents)

    def path(self) -> str:
        """按原文顺序描述覆盖范围。

        不能用 `best` 的路径：合并后正文是从 seq 最小的父块开始拼的，
        用最高分那条的标题当块名会名实不符 —— 模型看到「第四条」的标题，
        读到的却是「第三条」的正文，引用时就会张冠李戴。
        """
        paths = [p for _, _, p in self.parents if p]
        if not paths:
            return ""
        return paths[0] if len(paths) == 1 else f"{paths[0]} … {paths[-1]}"


def _key(evidence: Evidence) -> tuple[int, str, str]:
    row = evidence.row
    return (row["parent_seq"], row["parent_id"], row["section_path"])


def _group(evidence: list[Evidence], top_k: int) -> list[_Group]:
    """按 parent_id 保序去重，顺带合并同文档内相邻的父块。

    多个子块命中同一父块是常态（一个小节被切成 2~4 块，query 往往命中其中
    两块）。不去重就会让同一段条款在上下文里重复三遍，白烧 token 还挤占名额。
    """
    groups: list[_Group] = []
    seen_parents: set[str] = set()

    for item in evidence:
        parent = item.row["parent_id"]
        if parent in seen_parents:
            # 同一父块的第二个命中：只记 hit_chunk（截断时要用），不新建组
            owner = _parent_of(groups, parent)
            if owner is not None:
                owner.absorb(item)
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


def _parent_of(groups: list[_Group], parent_id: str) -> _Group | None:
    return next((g for g in groups if any(p == parent_id for _, p, _ in g.parents)), None)


# ── 回填与渲染 ────────────────────────────────────────────────────────────
def _render(group: _Group, window: bool = False) -> str:
    """把父块还原成原文。`window=True` 时只保留命中子块及其前后各一块。"""
    parts: list[str] = []
    for _, parent_id, _ in group.parents:  # 已按 parent_seq 有序
        rows = store.fetch_parent(parent_id)
        if window:
            rows = _hit_window(rows, group.hit_chunks)
        parts.extend(r["body"] for r in rows)
    assert len(parts) == len(set(parts)), f"证据内出现重复段落：group.parents={group.parents}"
    return "\n\n".join(parts)


def _hit_window(rows: list[dict], hit_chunks: set[str]) -> list[dict]:
    keep: set[int] = set()
    for i, row in enumerate(rows):
        if row["chunk_id"] in hit_chunks:
            keep.update({i - 1, i, i + 1})
    # 一个命中都没有（该父块是被相邻合并带进来的）：保留首块，别整个丢空
    return [r for i, r in enumerate(rows) if i in keep] or rows[:1]


def _to_section(group: _Group, text: str) -> PolicySection:
    row = group.best.row
    path = group.path()
    return PolicySection(
        section=f"{row['doc_id']} {row['title']}" + (f" > {path}" if path else ""),
        text=text,
        score=group.best.score,
        doc_id=row["doc_id"],
        layer=row["layer"],
        doc_type=row["doc_type"],
        effective_date=row["effective_date"],
        source_path=row["source_path"],
        reason=group.best.why(),
    )
