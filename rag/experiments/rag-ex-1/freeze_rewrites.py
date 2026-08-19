"""从一次评测结果提取 rewrite plan，供固定改写重跑使用。"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法：freeze_rewrites.py RESULT.json REWRITE_CACHE.json")
    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["cases"]
    cache = {
        row["query"]: row["rewrite_plan"]
        for row in rows
        if row.get("rewrite_plan")
    }
    Path(sys.argv[2]).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
