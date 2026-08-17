"""r1 自检 —— 在花钱跑批前拦住明显不对的样本（4-rag-dataset 8.1）。

    python rag/evals/validate_cases.py                  # 默认校验 rag/datasets/r1
    python rag/evals/validate_cases.py rag/datasets/r2

**不调模型**，只连 Milvus 查一次 ID。跑批一次要过完整检索链路加两个 LLM judge，
自检的成本是它的千分之一，所以每次改数据集、每次重新灌库都该先跑这个。

五项检查（对应 8.1 的五行）：

1. **`seed_chunk_id` 存在性**：ID 是位置派生的（`{doc_id}#{parent_seq}:{chunk_index}`），
   政策文档增删一个小节就会让整篇后续的 ID 静默偏移。失效时用例照跑，只是 Recall
   全线暴跌，看上去像检索退化 —— 这项检查把它变成一条明确的报错。
2. **重叠率**：口语档抄了原文的词，Recall 会虚高（5.1）。这项报警告不报错，
   超线的样本由人决定重写还是降级为 formal 档。
3. **参考答案可溯源**：答案里的数字必须在种子块正文出现过。凭空出现的数字是幻觉，
   而 Context Recall 会把它算成「检索没召回」，把生成的问题记到检索头上。
4. **`case_id` 与 query 去重**：措辞几乎相同的两条 query 会让同一类问题被重复计权。
5. **分层覆盖**：16 篇文档、两个 layer、两种 kind 都要有样本，冷门文档不低于下限。
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.evals.common import (  # noqa: E402
    DATASET_DIR,
    load_chunks,
    load_env,
    overlap_ratio,
    read_cases,
    terms,
)

MAX_COLLOQUIAL_OVERLAP = 0.45
"""口语档超过这个重叠率就报警告，口径与 generate_cases.py 一致。"""

MIN_CASES_PER_DOC = 4
"""每篇文档的样本下限 = 保底 2 个种子块 × 两档语域。"""

MIN_QUERY_DISTANCE = 0.15
"""两条 query 的实词差异下限。低于它视为重复问法。"""

STYLES = {"formal", "colloquial"}
TYPES = {"single", "multi_hop", "unanswerable"}
REQUIRED = ("case_id", "query", "style", "type", "seed_chunk_id", "reference_answer", "meta")

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


def _check_shape(rep: Report, case: dict) -> bool:
    cid = case.get("case_id", "<无 case_id>")
    missing = [f for f in REQUIRED if f not in case]
    if missing:
        rep.fail(cid, f"缺字段 {missing}")
        return False
    if case["style"] not in STYLES:
        rep.fail(cid, f"未知 style「{case['style']}」，可选 {sorted(STYLES)}")
    if case["type"] not in TYPES:
        rep.fail(cid, f"未知 type「{case['type']}」，可选 {sorted(TYPES)}")
        return False

    seeds = case["seed_chunk_id"]
    # 分母口径按 type 定（5-rag-eval 3.1）：single 是 1，multi_hop 是 2 且要求全中，
    # unanswerable 不进均值。种子数与 type 对不上，聚合出来的分数就是错的。
    expect = {"single": 1, "multi_hop": 2, "unanswerable": 0}[case["type"]]
    if len(seeds) != expect:
        rep.fail(cid, f"type={case['type']} 应有 {expect} 个 seed_chunk_id，实际 {len(seeds)}")
    if case["type"] == "unanswerable" and case["reference_answer"]:
        rep.fail(cid, "unanswerable 样本不该有参考答案 —— 语料里没有的事没有正确答案")
    if case["type"] != "unanswerable" and not case["reference_answer"].strip():
        rep.fail(cid, "参考答案为空，算不了 Context Recall")
    return True


def _check_traceable(rep: Report, case: dict, source: str) -> None:
    """答案里的数字必须在种子块正文里出现过。"""
    if not source:
        return
    # 正文里的数字带千分位或加粗标记，去掉非数字字符再比对，避免把 **15** 判成没出现
    pool = set(_NUMBER.findall(source))
    stray = [n for n in _NUMBER.findall(case["reference_answer"]) if n not in pool]
    if stray:
        rep.fail(case["case_id"], f"参考答案里的数字 {stray} 在种子块正文里找不到")


def validate(dataset: Path) -> tuple[Report, list[dict]]:
    rep = Report()
    cases = read_cases(dataset)
    if not cases:
        raise SystemExit(f"{dataset / 'cases.jsonl'} 里没有样本")

    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    all_docs = {c["doc_id"] for c in chunks}

    seen_ids: set[str] = set()
    seen_terms: list[tuple[str, set[str]]] = []
    per_doc: Counter[str] = Counter()
    layers: Counter[str] = Counter()
    kinds: Counter[str] = Counter()

    for case in cases:
        if not _check_shape(rep, case):
            continue
        cid = case["case_id"]

        # 1 · ID 存在性
        for chunk_id in case["seed_chunk_id"]:
            if chunk_id not in by_id:
                rep.fail(
                    cid,
                    f"种子块 {chunk_id} 不在 collection 里 —— 切片版本漂了，"
                    "重新灌库或重新生成受影响文档的样本",
                )

        # 2 · 重叠率
        source = "".join(by_id[c]["body"] for c in case["seed_chunk_id"] if c in by_id)
        if source:
            actual = overlap_ratio(case["query"], source)
            recorded = case["meta"].get("overlap_ratio", -1)
            if abs(actual - recorded) > 0.01:
                rep.fail(cid, f"meta.overlap_ratio 与实际不符：记的 {recorded}，实算 {actual}")
            if case["style"] == "colloquial" and actual > MAX_COLLOQUIAL_OVERLAP:
                rep.warn(cid, f"口语档重叠率 {actual} 超线，query 抄了原文的词：{case['query']}")

        # 3 · 参考答案可溯源
        _check_traceable(rep, case, source)

        # 4 · 去重
        if cid in seen_ids:
            rep.fail(cid, "case_id 重复")
        seen_ids.add(cid)
        mine = terms(case["query"])
        for other_id, other in seen_terms:
            union = mine | other
            if union and 1 - len(mine & other) / len(union) < MIN_QUERY_DISTANCE:
                rep.warn(cid, f"与 {other_id} 的问法几乎相同：{case['query']}")
        seen_terms.append((cid, mine))

        # 5 · 分层覆盖的计数
        for chunk_id in case["seed_chunk_id"]:
            if chunk := by_id.get(chunk_id):
                per_doc[chunk["doc_id"]] += 1
                layers[chunk["layer"]] += 1
                kinds[chunk["kind"]] += 1

    for doc_id in sorted(all_docs):
        if per_doc[doc_id] < MIN_CASES_PER_DOC:
            rep.fail(
                "分层覆盖",
                f"{doc_id} 只有 {per_doc[doc_id]} 条样本，低于下限 {MIN_CASES_PER_DOC}",
            )
    facets = (("layer", layers, {"law", "platform"}), ("kind", kinds, {"text", "table"}))
    for name, got, want in facets:
        if missing := want - set(got):
            rep.fail("分层覆盖", f"{name} 缺 {sorted(missing)}，这一档测不出来")

    return rep, cases


def _summary(cases: list[dict]) -> str:
    by_type = Counter(c["type"] for c in cases)
    by_style = Counter(c["style"] for c in cases)
    by_doc: dict[str, int] = defaultdict(int)
    for case in cases:
        by_doc[case["meta"]["doc_id"] or "—"] += 1
    ratios = sorted(
        c["meta"]["overlap_ratio"]
        for c in cases
        if c["style"] == "colloquial" and c["seed_chunk_id"]
    )
    median = ratios[len(ratios) // 2] if ratios else 0.0
    return (
        f"  类型：{dict(by_type)}\n"
        f"  语域：{dict(by_style)}\n"
        f"  口语档重叠率中位数：{median}\n"
        f"  文档：{'  '.join(f'{d}×{n}' for d, n in sorted(by_doc.items()))}"
    )


def main() -> None:
    dataset = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DATASET_DIR
    if not (dataset / "cases.jsonl").exists():
        raise SystemExit(f"找不到 {dataset / 'cases.jsonl'}")

    load_env()
    rep, cases = validate(dataset)

    print(f"数据集 {dataset.name}：{len(cases)} 条样本")
    print(_summary(cases))

    if rep.warnings:
        print(f"\n! {len(rep.warnings)} 条警告（人工决定改不改）：")
        for msg in rep.warnings:
            print(f"  - {msg}")
    if rep.errors:
        print(f"\n✗ {len(rep.errors)} 处不合格：")
        for msg in rep.errors:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("\n✓ ID 有效、答案可溯源、分层齐全")


if __name__ == "__main__":
    main()
