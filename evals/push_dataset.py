"""把 cases.jsonl 推成 Langfuse 数据集（2-design 6.1 的 ②）。

    python evals/push_dataset.py                    # evals/dataset/d1 → 数据集 refund-cases-d1
    python evals/push_dataset.py evals/dataset/d1 --name refund-cases-d1
    python evals/push_dataset.py --dry-run          # 只打印将要推的内容，不连 Langfuse

一条用例 = 一个 dataset item，**item id 直接用 case_id**：Langfuse 按 id upsert，
所以改完 cases.jsonl 重推是覆盖同一条，不会在数据集里堆出第二份。

一条用例被拆成三块，切分口径决定了报告好不好用：

| 字段 | 放什么 | 谁读它 |
|---|---|---|
| `input` | 喂给 Agent 的东西：`context` + 每轮的 user 文本 | 跑批时的 task 函数 |
| `expected_output` | 判分依据：每轮的 `expected`，原样搬过去 | 打分器 |
| `metadata` | 标题 / 标签 / 优先级 / 备注 | 人在 UI 上筛用例、看报告 |

推之前先跑 `python evals/validate_cases.py`——不自洽的用例推上去，报告里的红只会
误导人（2-design 6.2 第一层）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "dataset" / "d1"


def load_env(path: Path) -> None:
    """把 .env 里的键补进 os.environ（已有的环境变量优先，不覆盖）。

    run-main.sh 用 `source` 干这件事，评估脚本是直接 `python` 跑的，所以自己读一遍。
    只认 KEY=VALUE 并剥掉引号，变量展开这类 shell 语法不支持——.env 里也没用到。
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_cases(dataset: Path) -> list[dict]:
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


def to_item(case: dict) -> dict:
    # `run`（只有 D1-027 用到）按同一条口径拆开：怎么跑归 input，判什么归
    # expected_output —— 两边分清楚，打分器直接读 expected_output。
    payload = {
        "context": case["context"],
        "turns": [{"user": turn["user"]} for turn in case["turns"]],
    }
    expected = {"turns": [turn["expected"] for turn in case["turns"]]}
    if "run" in case:
        spec = dict(case["run"])
        expected["run"] = spec.pop("expected", {})
        payload["run"] = spec

    return {
        "id": case["case_id"],
        "input": payload,
        "expected_output": expected,
        "metadata": {
            "title": case["title"],
            "tags": case.get("tags", []),
            "priority": case.get("priority", ""),
            "turn_count": len(case["turns"]),
            "note": case.get("_note", ""),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="把 cases.jsonl 推成 Langfuse 数据集")
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET), help="数据集目录")
    parser.add_argument("--name", help="Langfuse 上的数据集名，默认 refund-cases-<目录名>")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dry-run", action="store_true", help="只打印，不连 Langfuse")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    cases = load_cases(dataset)
    name = args.name or f"refund-cases-{dataset.name}"
    items = [to_item(case) for case in cases]

    if args.dry_run:
        print(f"数据集 {name} ← {dataset}（{len(items)} 条）")
        print(json.dumps(items[0], ensure_ascii=False, indent=2))
        return

    load_env(Path(args.env_file))
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not (public_key and secret_key):
        raise SystemExit("缺 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，先填 .env")
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or None

    from langfuse import Langfuse

    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    if not client.auth_check():
        raise SystemExit(f"Langfuse 凭据校验失败：{host or '默认端点'}")

    # 数据集本身也按 name upsert；描述里带上绑定关系，免得日后看 UI 时
    # 不知道这批期望值是对着哪版 fixture 和规则引擎写的（见 dataset/d1/README.md）。
    client.create_dataset(
        name=name,
        description=f"RefundAgent 离线回归用例集，来自 {dataset.relative_to(ROOT)}/cases.jsonl",
        metadata={"source_dir": str(dataset.relative_to(ROOT)), "case_count": len(items)},
    )
    for item in items:
        client.create_dataset_item(dataset_name=name, **item)
    client.flush()

    print(f"已推送 {len(items)} 条 → 数据集 {name}（{host or 'cloud.langfuse.com'}）")


if __name__ == "__main__":
    sys.exit(main())
