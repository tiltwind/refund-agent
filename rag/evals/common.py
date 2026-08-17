"""r1 数据集三个脚本的公共部分：环境变量、语料、jsonl、实词重叠率。

语料从 **Milvus 取**，不从 `doc/policy/` 重新切一遍。数据集绑的是 `chunk_id`，
而 `chunk_id` 由切片位置派生（rag/chunking/policy.py），脚本自己再切一次就等于
在库外维护第二份切片产物：切分参数一改，两边悄悄错开，生成时看着对、评测时全线
判负。collection 是唯一事实源，生成与自检读的都是它。
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "rag" / "datasets" / "r1"

# query 与块正文比对时忽略的虚词。只在两个字都是虚词时才丢掉这个二元组 ——
# 「的商品」里的「商品」还是实词。
FUNC_CHARS = set("的了着过吗呢吧啊呀么是不在有和与或及对为以之其这那些我你他她它们个就都还也很要会能可把被给让从到")

_TOKEN = re.compile(r"[0-9]+(?:\.[0-9]+)?|[a-zA-Z]+|[一-鿿]+")


def load_env(path: Path | None = None) -> None:
    """把 .env 里的键补进 os.environ（已有的环境变量优先）。

    run-main.sh 用 source 干这件事，评估脚本是直接 python 跑的，所以自己读一遍。
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_chunks() -> list[dict]:
    """取 collection 里的全部子块，按 chunk_id 升序。

    升序是硬要求：抽样在这个序列上跑随机数，顺序变了同一个随机种子会抽出另一批块。
    """
    from rag.retrieving import store

    rows = store.client().query(
        collection_name=store.COLLECTION,
        filter="chunk_index >= 0",
        output_fields=store.FIELDS,
        limit=16384,
    )
    if not rows:
        raise SystemExit(
            f"collection「{store.COLLECTION}」是空的（uri={store.MILVUS_URI}）；"
            "先执行 python rag/index/seed_milvus.py 灌库"
        )
    return sorted(rows, key=lambda r: r["chunk_id"])


# ── 实词重叠率 ────────────────────────────────────────────────────────────
def terms(text: str) -> set[str]:
    """把文本切成可比对的实词单位：中文取相邻二元组，数字与拉丁词整体取。

    这里**没有分词**：项目不装 jieba，BM25 的中文分析器跑在 Milvus 服务端
    （rag/index/seed_milvus.py），本机拿不到同一套切词。二元组是零依赖下最接近
    中文词的近似 —— 「无理由退货」切出「无理」「理由」「由退」「退货」，
    抄了原文的 query 会大面积命中，换过说法的不会。

    它只用于**分档与打回重写**，不参与判分，所以近似带来的误差可以接受；
    真要用它做门禁，就得先跟服务端分析器对齐。
    """
    out: set[str] = set()
    for token in _TOKEN.findall(text):
        if token[0].isascii():
            out.add(token.lower())
            continue
        if len(token) == 1:
            if token not in FUNC_CHARS:
                out.add(token)
            continue
        for a, b in zip(token, token[1:]):
            if a in FUNC_CHARS and b in FUNC_CHARS:
                continue
            out.add(a + b)
    return out


def overlap_ratio(query: str, body: str) -> float:
    """query 有多少实词单位来自块正文。1.0 = 整句都是原文的词。"""
    q = terms(query)
    if not q:
        return 0.0
    return round(len(q & terms(body)) / len(q), 3)


# ── jsonl ────────────────────────────────────────────────────────────────
def read_cases(dataset: Path) -> list[dict]:
    rows = []
    with (dataset / "cases.jsonl").open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"cases.jsonl 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return rows


def write_cases(dataset: Path, cases: list[dict]) -> None:
    dataset.mkdir(parents=True, exist_ok=True)
    with (dataset / "cases.jsonl").open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
