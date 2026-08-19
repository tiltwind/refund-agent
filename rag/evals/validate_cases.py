"""r1 自检 —— 在花钱跑批之前拦住明显不对的样本，不调模型。

    python rag/evals/validate_cases.py

五项检查，对着数据集的四个字段：

| 检查 | 抓什么 |
|---|---|
| 字段齐全、`case_id` 唯一 | 跑批时才发现要重来 |
| `question` 不重复 | 同一个问题重复计权 |
| `source` 里的 chunk_id 在库里存在 | 切片版本漂了（4 · 五） |
| `ground_truth` 里的数字在源块正文里出现过 | 凭空出现的天数/次数/金额是模型幻觉 |
| `ground_truth` 不以独立结论句开头 | 「可以退。」切出来是不含信息的判定单元 |

第三项只查存在性 —— `source` 不参与判分，但它失效说明数据集绑的那版切片已经
不在了，标准答案大概率也对不上了。
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.evals.common import DATASET_DIR, load_chunks, load_env, read_cases  # noqa: E402

FIELDS = ("case_id", "question", "ground_truth", "source")

LEAD_MAX_CHARS = 8
"""首句短于这个长度就当独立结论句。Context Recall 把 `ground_truth` 按句拆开
逐句归因，「可以退。」这样的句子不含条件、期限、数字，检回的条款再对也支撑
不了它，判负记在检索头上 —— 结论要连着它管的前提一起写。"""

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_END = re.compile(r"(?<=[。！？；])")


def check(cases: list[dict], bodies: dict[str, str]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}

    for i, case in enumerate(cases, 1):
        if missing := [f for f in FIELDS if not case.get(f)]:
            errors.append(f"第 {i} 行缺字段：{'、'.join(missing)}")
            continue

        cid = case["case_id"]
        if cid in seen_ids:
            errors.append(f"{cid}：case_id 重复")
        seen_ids.add(cid)

        question = case["question"].strip()
        if first := seen_questions.get(question):
            errors.append(f"{cid}：question 与 {first} 完全相同")
        seen_questions[question] = cid

        parts = [p.strip() for p in _SENTENCE_END.split(case["ground_truth"]) if p.strip()]
        if len(parts) > 1 and len(parts[0]) <= LEAD_MAX_CHARS:
            errors.append(f"{cid}：ground_truth 以独立结论句「{parts[0]}」开头")

        if unknown := [sid for sid in case["source"] if sid not in bodies]:
            errors.append(f"{cid}：source 在库里不存在 {unknown} —— 切片版本漂了")
            continue

        # 数字可溯源：标准答案里的天数、次数、金额必须在源块正文里出现过。
        # 「7 天」写成「七天」不算漏 —— 中文数字不在这条检查的范围里，它抓的是
        # 模型凭空编出一个阿拉伯数字的情况
        source_text = "".join(bodies[sid] for sid in case["source"])
        source_numbers = set(_NUMBER.findall(source_text))
        if invented := sorted(set(_NUMBER.findall(case["ground_truth"])) - source_numbers):
            errors.append(f"{cid}：ground_truth 里的数字 {invented} 在源块正文里没有")

    return errors


def main() -> None:
    load_env()
    cases = read_cases(DATASET_DIR)
    bodies = {c["chunk_id"]: c["body"] for c in load_chunks()}
    print(f"{DATASET_DIR.name}：{len(cases)} 条样本，语料 {len(bodies)} 块")

    errors = check(cases, bodies)
    if not errors:
        print("零错误。下一步：python rag/experiments/rag-ex-1/run_experiment.py")
        return

    print(f"\n{len(errors)} 个错误：")
    for line in errors:
        print(f"  {line}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
