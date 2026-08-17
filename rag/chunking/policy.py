"""切分编排 —— 一篇政策 Markdown → 一串可入库的子块。

## 策略与理由

```
frontmatter          → 文档级元数据（过滤与重排用，不进向量）
按标题层级切          → 父块 = 一个标题路径下的正文（≈ 一条完整规则）
  父块内按段落切      → 子块 = 检索单元，目标 320 / 硬上限 512 token
    表格、代码        → 原子块，任何情况下不切
    超长自然段        → 语义切分兜底（semantic.py）
overlap = 0
每个子块加块头        → 【文档】+【路径】
```

**为什么 overlap = 0**：overlap 是用来防止「关键句正好落在切分边界上被劈成
两半」的，它的前提是切分边界与语义边界无关。这里恰恰相反 —— 边界是标题和
自然段，本身就是语义边界。在这种语料上加 overlap，换来的只是索引体积按比例
上涨、top-k 里出现内容高度重复的相邻块，没有召回收益。与其加 overlap，
不如把预算花在块头上（model.py 的 `header`），信息密度高得多。

**为什么父块不单独存**：父块 = 同 parent_id 的子块按序拼接，检索期还原
（rag/retrieving/pipeline/assemble.py）。overlap = 0 让这个还原是精确的。

## 参数

`320 / 512` 不是从 BGE-M3 的 8192 上限倒推的，是从**查询粒度**来的：
退款判定的问题几乎全是事实型（「签收 10 天还能退吗」「拆封了还能退吗」），
答案就是一两句话。这类查询的理想检索单元在 256~400 token —— 块再大，
命中的块里无关内容占比上升，向量被摊平，反而排不上去。上下文完整性由父块
回填负责，不靠把子块做大。
"""

import os
from collections.abc import Callable
from pathlib import Path

from rag.chunking.markdown import parse_frontmatter, split_blocks, split_sections
from rag.chunking.model import Block, Chunk
from rag.chunking.semantic import semantic_split

CHILD_TARGET_TOKENS = int(os.getenv("POLICY_CHUNK_TARGET", "320"))
CHILD_MAX_TOKENS = int(os.getenv("POLICY_CHUNK_MAX", "512"))

Tokens = Callable[[str], int]
Encode = Callable[[list[str]], list[list[float]]]


def chunk_document(path: Path, root: Path, tokens: Tokens, encode: Encode) -> list[Chunk]:
    """把一篇政策文档切成子块。`root` 用于算仓库内相对路径（做引用定位）。"""
    rel = str(path.relative_to(root))
    doc, body = parse_frontmatter(path.read_text(encoding="utf-8"), rel)

    chunks: list[Chunk] = []
    for seq, group in enumerate(_group_by_top_heading(split_sections(body))):
        parent_id = f"{doc.doc_id}#{seq:03d}"
        idx = 0
        for section in group:
            for block in _pack(split_blocks(section.text), tokens, encode):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}:{idx:02d}",
                        parent_id=parent_id,
                        chunk_index=idx,
                        doc=doc,
                        section_path=section.path,
                        body=block.text,
                        kind=block.kind,
                        parent_seq=seq,
                    )
                )
                idx += 1
    return chunks


def _group_by_top_heading(sections: list[Section]) -> list[list[Section]]:
    """把 section 按**顶层标题**（`##` 级）归组，一组 = 一个父块。

    父块粒度定在这一层，是因为下一层（`###`）切得太细：法规层一个 `###`
    就是一条 100 token 的条文，如果父块也定在这里，父子块会退化成 1:1 ——
    「小块检索、大块喂模型」这套东西就白做了，回填回来的还是那一小块。

    定在 `##` 之后，父块是一章 / 一条完整规则（数百 token），子块仍是条文级。
    L02「第三章 退货程序」下的五个条文各自可被精确命中，而命中任一条都能把
    整章程序回填给模型 —— 事实型查询靠子块的精度，操作型查询靠父块的完整性。

    分组按「路径的第一段」而非固定层级，因此对 L04 / L05 这类 `#` 直接跳到
    `###` 的文档，父块会退化成一条条文（子:父 = 1:1），回填拿不到更多上下文。
    这里**不做补救** —— 索引期按大小硬凑父块是在猜哪几条该在一起。真正需要
    上下文时由装配阶段处理：命中的相邻父块在那里按需合并
    （rag/retrieving/pipeline/assemble.py），合并的依据是「这次确实都命中了」，
    比索引期的猜测可靠。
    """
    groups: list[list[Section]] = []
    current_key: str | None = None
    for section in sections:
        # 空 path = frontmatter 与第一个 `##` 之间的文档引言，自成一组
        key = section.path[0] if section.path else f"__preamble__{section.seq}"
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(section)
    return groups


def _pack(blocks: list[Block], tokens: Tokens, encode: Encode) -> list[Block]:
    """把段落级 block 合并成目标大小的子块。

    三条规则，优先级从高到低：
    1. 原子块（表格 / 代码）永不切开，哪怕它自己就超上限；
    2. 超长自然段交给语义切分；
    3. 其余相邻段落累加到 target 就收尾。
    """
    out: list[Block] = []
    cur: list[Block] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            out.append(_merge(cur))
            cur, cur_tokens = [], 0

    for block in blocks:
        n = tokens(block.text)

        if block.atomic:
            # 表格前通常有一句「按以下顺序审核」的引子，让它跟表格待在一起 ——
            # 除非合并会撑破硬上限，那就先把引子收尾。
            if cur and cur_tokens + n > CHILD_MAX_TOKENS:
                flush()
            cur.append(block)
            cur_tokens += n
            # 原子块本身超上限（大表）时也不切，整块单独成一个子块
            if cur_tokens >= CHILD_TARGET_TOKENS:
                flush()
            continue

        if n > CHILD_MAX_TOKENS:
            flush()
            out.extend(
                Block("text", piece)
                for piece in semantic_split(
                    block.text, encode, tokens, CHILD_TARGET_TOKENS, CHILD_MAX_TOKENS
                )
            )
            continue

        if cur and cur_tokens + n > CHILD_TARGET_TOKENS:
            flush()
        cur.append(block)
        cur_tokens += n

    flush()
    return out


def _merge(blocks: list[Block]) -> Block:
    # 只要混进了表格或代码，整块就按原子块看待 —— 下游（装配时的截断）
    # 据此避免从表格中间下刀
    kind = next((b.kind for b in blocks if b.atomic), "text")
    return Block(kind, "\n\n".join(b.text for b in blocks))
