"""给已有样本补上 `claims`，直接写回 `cases.jsonl`。

    python rag/evals/generate_claims.py                 # 只补缺的
    python rag/evals/generate_claims.py --force         # 全部重拆
    python rag/evals/generate_claims.py --force --cases R1-011 R1-012   # 只重拆这几条
    python rag/evals/generate_claims.py --limit 3       # 调提示词用
    python rag/evals/generate_claims.py --dry-run       # 只打印要拆哪些，不调模型

`claims` 是 `reference_answer` 拆成的原子事实，Context Recall 逐条判它有没有被检回的
上下文支撑，分母就是这里的条数（5-rag-eval 七）。

**新样本不用这个脚本**：`generate_cases.py` 生成 query 和参考答案时一并出 claim，一次
调用就出齐。这个脚本给的是已经生成好的样本 —— 重跑 generate_cases 会连 query 一起重写，
措辞漂了，整个数据集的历史分数就不可比。

## 为什么提前拆、而不是跑批时现拆

1. **拆 claim 与被测链路无关**，只依赖 `reference_answer`。跑一次批拆 96 遍，改一版
   检索参数再拆 96 遍，花的钱全是重复的。
2. **分母必须稳定**。同一条答案这次拆 4 条、下次拆 5 条，Context Recall 的分子分母
   一起变，两次 run 的分数没法比 —— 而版本对比正是这个指标的用途。温度 0 也挡不住
   这个抖动（改写那一步已经实测到）。

同一份参考答案只拆一次：`formal` 与 `colloquial` 是同一个种子块的两种问法，参考答案
逐字相同，共用一份 claim。调模型的次数少一半，两条样本的分母也保持一致。

补拆用的是 judge 模型（`OPENAI_JUDGE_MODEL` 那档），不是生成样本的模型，所以补出来的
行记一个 `meta.claims_by` —— 同一个数据集里 claim 的来源不止一处，得看得出来。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field  # noqa: E402

from rag.evals.common import (  # noqa: E402
    DATASET_DIR,
    build_judge,
    judge_json_hint,
    judge_name,
    load_env,
    read_cases,
    write_cases,
)

MAX_CLAIMS = 6
"""一条参考答案最多拆几条，与 `generate_cases.py` 的提示词同口径。答案本身只有一两百字，
拆到更多说明模型在把一句话切成半句半句 —— 那种 claim 判定起来全是「部分支撑」。"""


class Claims(BaseModel):
    claims: list[str] = Field(description="参考答案里的原子事实，每条一句，按答案中出现的顺序")


SYSTEM = f"""你在处理退款政策评测数据。把一段参考答案拆成若干条**原子事实**（claim）。

1. 每条 claim 只说一件事：一个条件、一个期限、一个金额、一项义务或一个例外。
2. 每条必须**独立可读**：不用「它」「该商品」「上述情形」这类指代，把主语和前提写进句子里。
   ✓「拆封后的音像制品不适用七天无理由退货」
   ✗「这类商品也不适用」
3. 数字、天数、比例、金额留在所属的那条里，不单独拆成一条，也不要丢掉。
4. **只用给定答案里的信息**。不补常识、不做推理、不加「建议消费者…」这类答案里没有的话。
   答案说错了也照拆 —— 这一步不判对错。
5. 「也就是说」「因此」这类连接词不单独成条；但它后面如果给出了一个新的判断
   （「这本身即构成违法」），那是独立的一条。
6. 答案开头那种一句话结论（「可以退。」「不给退。」「要收费。」）**不许原样成条** ——
   它离开答案就读不懂。把它管的那个前提写进去：
   ✓「消费者基于查验需要拆封的，商品仍可退货」
   ✗「可以退。」
7. 最多 {MAX_CLAIMS} 条。答案很短就只出一两条，不要为了凑数把一句话切碎。"""


def needs_claims(case: dict) -> bool:
    """`unanswerable` 没有参考答案，也就没有 Context Recall 可算（分母是 0）。"""
    return case["type"] != "unanswerable" and bool(case["reference_answer"].strip())


def split(model, answer: str) -> list[str]:
    result = model.invoke(
        [
            {"role": "system", "content": SYSTEM + judge_json_hint(Claims)},
            {"role": "user", "content": answer},
        ]
    )
    return [c.strip() for c in result.claims if c.strip()][:MAX_CLAIMS]


def main() -> None:
    parser = argparse.ArgumentParser(description="给已有样本补 claims，写回 cases.jsonl")
    parser.add_argument("dataset", nargs="?", default=str(DATASET_DIR), help="数据集目录")
    parser.add_argument("--force", action="store_true", help="全部重拆，忽略已有的 claims")
    parser.add_argument("--cases", nargs="*", help="只处理这些 case_id，配 --force 用来重拆某几条")
    parser.add_argument("--limit", type=int, help="只处理前 N 条待拆样本，调提示词用")
    parser.add_argument("--dry-run", action="store_true", help="只打印要拆哪些，不调模型")
    args = parser.parse_args()

    load_env()
    dataset = Path(args.dataset).resolve()
    cases = read_cases(dataset)
    wanted = set(args.cases) if args.cases else None
    todo = [
        c for c in cases
        if needs_claims(c) and (args.force or not c.get("claims"))
        and (not wanted or c["case_id"] in wanted)
    ]
    if args.limit:
        todo = todo[: args.limit]

    print(f"→ {dataset.name}：{len(cases)} 条样本，待拆 {len(todo)} 条")
    if args.dry_run:
        for case in todo:
            print(f"  {case['case_id']}  {case['reference_answer'][:60]}…")
        return
    if not todo:
        print("  claims 都齐了，无需调模型")
        return

    model = build_judge(Claims, max_retries=2)
    print(f"  judge：{judge_name()}（温度 0）")

    # 参考答案逐字相同的样本共用一份 claim，见模块 docstring
    by_answer: dict[str, list[str]] = {}
    if not args.force:
        for case in cases:
            if case.get("claims"):
                by_answer.setdefault(case["reference_answer"].strip(), case["claims"])

    failed = []
    for i, case in enumerate(todo, 1):
        answer = case["reference_answer"].strip()
        if reuse := by_answer.get(answer):
            case["claims"] = list(reuse)
            case["meta"]["claims_by"] = judge_name()
            print(f"  [{i}/{len(todo)}] {case['case_id']} → {len(reuse)} 条（与同答案样本复用）")
            continue
        try:
            claims = split(model, answer)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{case['case_id']}: {type(exc).__name__}: {exc}")
            print(f"  [{i}/{len(todo)}] {case['case_id']} 失败：{type(exc).__name__}")
            continue
        if not claims:
            failed.append(f"{case['case_id']}: 模型返回 0 条 claim")
            continue
        by_answer[answer] = claims
        case["claims"] = claims
        case["meta"]["claims_by"] = judge_name()
        print(f"  [{i}/{len(todo)}] {case['case_id']} → {len(claims)} 条")

    write_cases(dataset, cases)
    have = sum(1 for c in cases if c.get("claims"))
    total = sum(len(c.get("claims") or []) for c in cases)
    print(f"\n  已写回 {dataset / 'cases.jsonl'}：{have} 条样本 / {total} 条 claim")
    if failed:
        print(f"  {len(failed)} 条没拆出来，重跑本脚本会补：")
        for line in failed:
            print(f"    {line}")


if __name__ == "__main__":
    main()
