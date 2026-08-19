"""Markdown 结构解析 —— frontmatter、标题树、段落/表格/代码块。

**结构感知优先于任何「聪明」的算法。** doc/policy 下的文档全部按「第 X 条」
组织标题，标题本身就是最好的切分边界 —— 强行用固定长度或纯语义切分把
「第二条 退货窗口」和「第三条 商品条件」黏在一起，损失远大于块大小不理想。
"""

import re

import yaml

from rag.chunking.model import Block, DocMeta, Section

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_HR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|")


# ── frontmatter ───────────────────────────────────────────────────────────
def parse_frontmatter(raw: str, source_path: str) -> tuple[DocMeta, str]:
    """拆出 YAML frontmatter 与正文。缺 frontmatter 直接失败。

    元数据不是可选装饰：检索靠 effective_date / expire_date 排除已废止的条款，
    靠 layer / authority_level 在法规与平台规则冲突时定序。缺元数据的政策文档
    入库后，会以「看起来正常」的方式污染每一次检索。
    """
    m = _FRONTMATTER.match(raw)
    if not m:
        raise ValueError(f"{source_path}：缺少 YAML frontmatter，无法确定生效日期与效力位阶")

    meta = yaml.safe_load(m.group(1)) or {}
    missing = [k for k in ("doc_id", "title", "layer", "effective_date", "expire_date") if not meta.get(k)]
    if missing:
        raise ValueError(f"{source_path}：frontmatter 缺少必填字段 {missing}")

    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    doc = DocMeta(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        layer=str(meta["layer"]),
        category=str(meta.get("category", "")),
        doc_type=str(meta.get("doc_type", "")),
        # 法规写 authority（发布机关），平台政策写 publisher，这里归一
        authority=str(meta.get("authority") or meta.get("publisher") or ""),
        effective_date=str(meta["effective_date"]),
        expire_date=str(meta["expire_date"]),
        authority_level=int(meta.get("authority_level", 3)),
        version=str(meta.get("version") or meta.get("revision") or ""),
        retrieval_scope=str(meta.get("retrieval_scope", "")),
        tags=tuple(str(t) for t in tags),
        source_path=source_path,
    )
    return doc, raw[m.end() :]


# ── 标题树 ────────────────────────────────────────────────────────────────
def split_sections(body: str) -> list[Section]:
    """按标题层级切成 section，每个 section = 「某个标题路径下的**直接**正文」。

    这个定义自动处理了两种情况，不需要显式判断谁是叶子：
    - 无下级标题的「第三条」→ 它的正文成为一个 section；
    - 有下级标题的「第二章」→ 它自己的引言段成为一个 section，各下级标题
      各自再成 section。

    `#` 一级标题不进路径 —— 它与 frontmatter 的 title 重复，块头里已经有了。
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False
    seq = 0

    def flush() -> None:
        nonlocal seq, buf
        text = _clean("\n".join(buf))
        buf = []
        if not text:
            return
        sections.append(
            Section(seq=seq, path=tuple(t for lvl, t in stack if lvl > 1), text=text)
        )
        seq += 1

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        # 代码块里的 # 是注释，不是标题
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _clean(text: str) -> str:
    """去掉分节用的水平线与首尾空白。水平线只是排版，进了块就是纯噪声。"""
    kept = [ln for ln in text.splitlines() if not _HR.match(ln)]
    return "\n".join(kept).strip()


# ── 段落 / 表格 / 代码 ────────────────────────────────────────────────────
def split_blocks(text: str) -> list[Block]:
    """把一个 section 的正文切成段落级 block。

    Markdown 的「一个自然段」不等于「两个空行之间」：表格与列表的相邻行之间
    本来就没有空行。所以这里按类型分别收敛，而不是简单 `split("\\n\\n")`。
    """
    blocks: list[Block] = []
    lines = text.splitlines()
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # 代码块：以围栏配对，中间任何内容都不解析
        if _FENCE.match(line):
            fence = _FENCE.match(line).group(1)
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(Block("code", "\n".join(buf).strip()))
            continue

        # 表格：连续的 | 行（含 |---|---| 分隔行）整体成块
        if _TABLE_ROW.match(line):
            buf = []
            while i < n and _TABLE_ROW.match(lines[i]):
                buf.append(lines[i])
                i += 1
            blocks.append(Block("table", "\n".join(buf).strip()))
            continue

        # 普通段落：直到空行、表格行或代码围栏为止
        buf = []
        while i < n and lines[i].strip() and not _TABLE_ROW.match(lines[i]) and not _FENCE.match(lines[i]):
            buf.append(lines[i])
            i += 1
        blocks.append(Block("text", "\n".join(buf).strip()))

    return [b for b in blocks if b.text]
