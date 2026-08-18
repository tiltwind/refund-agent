"""把 r1 推成 Langfuse 数据集。

    python rag/evals/push_dataset.py                     # rag/datasets/r1 → retrieval-cases-r1
    python rag/evals/push_dataset.py --dry-run           # 只打印，不连 Langfuse

与 Agent 用例集的推送脚本（evals/push_dataset.py）分开：那边一条 item 是一段多轮
对话，这边一条 item 是一个 query。切分口径也不同 ——

| 字段 | 放什么 | 谁读它 |
|---|---|---|
| `input` | `query`，直接喂 `search_policy` | 跑批的 task 函数 |
| `expected_output` | `seed_chunk_id` + `reference_answer` + `claims` | 两个 Recall 打分器 |
| `metadata` | style / type / doc_id / layer / kind / 重叠率 | 报告分档（5-rag-eval 8.2） |

item id 直接用 `case_id`，Langfuse 按 id upsert：改完 cases.jsonl 重推是覆盖同一条。
推之前先跑 `python rag/evals/validate_cases.py`。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.evals.common import DATASET_DIR, ROOT, load_env, read_cases  # noqa: E402


def to_item(case: dict) -> dict:
    return {
        "id": case["case_id"],
        "input": {"query": case["query"]},
        "expected_output": {
            "seed_chunk_id": case["seed_chunk_id"],
            "reference_answer": case["reference_answer"],
            # claim 一起推：dataset run 里判 Context Recall 时读的是 item 自己带的这份，
            # 不回头读本地文件，两边不一致的风险就没了
            "claims": case.get("claims") or [],
        },
        "metadata": {"style": case["style"], "type": case["type"], **case["meta"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="把检索评测数据集推成 Langfuse 数据集")
    parser.add_argument("dataset", nargs="?", default=str(DATASET_DIR), help="数据集目录")
    parser.add_argument("--name", help="Langfuse 上的数据集名，默认 retrieval-cases-<目录名>")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不连 Langfuse")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    cases = read_cases(dataset)
    name = args.name or f"retrieval-cases-{dataset.name}"
    items = [to_item(case) for case in cases]

    if args.dry_run:
        print(f"数据集 {name} ← {dataset}（{len(items)} 条）")
        print(json.dumps(items[0], ensure_ascii=False, indent=2))
        return

    load_env()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not (public_key and secret_key):
        raise SystemExit("缺 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，先填 .env")
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or None

    from langfuse import Langfuse

    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    if not client.auth_check():
        raise SystemExit(f"Langfuse 凭据校验失败：{host or '默认端点'}")

    # 描述里带上 collection：期望值绑的是这一版切片产物，换了库历史分数不可比
    client.create_dataset(
        name=name,
        description=f"检索评测样本，来自 {dataset.relative_to(ROOT)}/cases.jsonl",
        metadata={
            "source_dir": str(dataset.relative_to(ROOT)),
            "case_count": len(items),
            "collection": os.getenv("MILVUS_COLLECTION", "refund_policy_chunks"),
        },
    )
    for item in items:
        client.create_dataset_item(dataset_name=name, **item)
    client.flush()

    print(f"已推送 {len(items)} 条 → 数据集 {name}（{host or 'cloud.langfuse.com'}）")


if __name__ == "__main__":
    main()
