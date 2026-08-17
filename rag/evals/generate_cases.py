"""r1 生成脚本 —— 分层抽样 + 反向生成 + 重叠率（4-rag-dataset 四 / 五 / 六）。

    python rag/evals/generate_cases.py                  # 生成 rag/datasets/r1/cases.jsonl
    python rag/evals/generate_cases.py --dry-run        # 只打印抽样结果，不调模型
    python rag/evals/generate_cases.py --limit 5        # 只生成前 5 个种子块，调提示词用

反向生成：从块出发写 query，`seed_chunk_id` 在生成时就确定，不需要在 353 个块里
人工穷举「这条 query 的全部相关块」。代价是这个 ID 不完整（同一条规则散在多篇
文档里），所以它只作 Recall 的下界，长尾由 Context Recall 覆盖。

**重复跑要出同一批样本**：随机种子固定、抽样在按 chunk_id 排序后的序列上跑、
生成温度 0。否则每次跑出来的数据集不同，版本之间的分数没法比。种子块选中之后，
query 仍由模型生成，措辞可能有细微出入 —— 要的是抽样稳定，不是逐字节可复现。
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field  # noqa: E402

from rag.evals.common import (  # noqa: E402
    DATASET_DIR,
    load_chunks,
    load_env,
    overlap_ratio,
    terms,
    write_cases,
)

SEED = 20260817
"""抽样随机种子。改它等于换一批种子块，整个 r1 要重跑重抽检。"""

BASE_QUOTA = 2
"""每篇文档的保底名额。16 篇都要有样本 —— 重排里 P02 有 1.0 的文档先验、其余 0.9，
不覆盖冷门文档就测不出这个先验有没有压死长尾。"""

BONUS_MIN_CHUNKS = 20
"""块数到这个量的文档多给一个名额。只加 1 不按比例分：按比例分会让 L02（41 块）
拿走的名额是 P02（10 块）的四倍，而数据集的作用是暴露问题，不是复现命中分布。"""

MIN_BODY_CHARS = 30
"""正文短于此的块不作种子：「以下行为构成对售后规则的滥用：」这类引子句
生成不出有答案的问题。它们仍在库里、仍可能被召回，只是不作为真值锚。"""

MAX_COLLOQUIAL_OVERLAP = 0.45
"""口语档的重叠率上线，超了打回重写一次（5.1）。formal 档不设限 —— 那一档本来
就是术语问法，它的分数读作上界。"""

MULTI_HOP = [
    ("拆封了能不能退、运费谁出", ("P05", "商品状态 划分 完好 拆封"), ("P06", "无理由退货 运费 承担")),
    ("会员的退货窗口和普通会员差多少", ("P07", "无理由退货窗口 会员等级"), ("P02", "退货窗口 无理由 签收")),
    ("生鲜坏了到底能不能退", ("P04", "生鲜 易腐 法定不适用"), ("P11", "生鲜食品 退款限制")),
    ("用了券的订单退多少钱、什么时候到账", ("P10", "优惠券 满减 分摊 退款金额"), ("P03", "到账时效 平台承诺")),
    ("平台的七天口径和法规原文差在哪", ("P02", "退货窗口 无理由 签收"), ("L02", "七日无理由退货 期限 收到商品")),
    ("被判成高风险账户之后还退不退得了", ("P08", "高风险账户 判定 标准"), ("P02", "判定顺序 审核 风控")),
    ("质量有问题该谁举证、怎么修换退", ("L05", "修理 更换 退货 期限"), ("L01", "瑕疵 举证 经营者")),
    ("平台不给退、我还能找谁", ("P09", "平台介入 申请 条件"), ("L04", "平台 协助维权 责任")),
]
"""跨块样本：两个块一起喂给生成器，`seed_chunk_id` 记两个（5.4）。

关键词只用于**从两篇文档里各挑一个块**，不进提示词 —— 挑块的口径必须与生成
无关，否则等于把答案提前泄露给了生成器。"""

UNANSWERABLE = [
    ("保价", "我前脚刚买后脚就降价了，你们保价吗，差价能退我不？"),
    ("以旧换新", "旧机器抵扣的那部分钱，退货的时候是退现金还是退回抵扣额度？"),
    ("延长保修", "想再买两年的延保服务，多少钱，怎么买？"),
    ("跨境关税", "海外直邮那单退货，当初交的关税能一起退回来吗？"),
    ("门店自提", "我是到线下门店自提的，能直接拿去门店退掉吗？"),
    ("直播打赏", "直播间一时冲动打赏出去的钱，能申请退回来吗？"),
]
"""语料里根本没有的事（5.4）。检索链路对空结果的口径是直接抛异常
（rag/retrieving/milvus.py），这类样本用来确认兜底行为，不进前三个指标的均值。"""

SYSTEM = """你在为退款政策检索系统构造评测样本。给你一段政策条文，你写出用户会问的问题和参考答案。

硬约束：
1. 两个问题问的是同一件事，只是语域不同：
   - formal：带政策术语的规范问法，一句完整问句。
   - colloquial：真实消费者的口语，可以带情绪、可以啰嗦、**不许出现条文里的专有措辞**
     （「无理由退货」「二次销售」「生效」这类），换成大白话说同一件事。
     ✓「拆开包装看了一眼，没用过，这种还能退吗？」
     ✗「已开启包装但未使用的商品是否适用无理由退货？」
2. 两个问题都必须能被这段条文回答，且问的是条文里最具体的那条规则，不是它所属的大话题。
   ✗「退款要多久」这种谁都能答的泛问。
3. reference_answer 只能用给定条文里的信息，一到三句话说完。不许补常识、不许编数字、
   不许写条文里没有的条件。条文没说的就不要说。
4. 不要在问题里引用条款号、文档名或标题。"""

USER_SINGLE = """政策条文：

{body}"""

USER_MULTI = """下面是两段来自不同文档的政策条文。写出的问题必须**两段都用上才答得全**，
只看其中一段只能答一半。参考答案要把两段的信息都写进去。

条文一：

{body_a}

条文二：

{body_b}"""

RETRY = """

注意：下面这个 colloquial 问法不合格，它直接搬了条文里的措辞（实词重叠率 {ratio}）：
「{rejected}」
换一种说法问同一件事，用日常口语，把条文里的名词换成用户自己会用的说法。"""


class Generated(BaseModel):
    formal: str = Field(description="带政策术语的规范问法")
    colloquial: str = Field(description="消费者的口语问法，不含条文措辞")
    reference_answer: str = Field(description="只用给定条文信息作答，一到三句")


# ── 抽样 ──────────────────────────────────────────────────────────────────
def _usable(chunk: dict) -> bool:
    """能不能当种子块。三类排除掉的块仍在库里、仍可能被召回，只是不作真值锚。"""
    body = chunk["body"].strip()
    if len(body) < MIN_BODY_CHARS:
        return False
    # 「……：」结尾的短块是后面表格或列表的引子，本身不含规则
    if len(body) < 80 and body.endswith("："):
        return False
    # section_path 为空 = 文档引言（「本清单分三级…」这类说明），它讲的是文档本身
    # 不是规则，反向生成只能造出「这篇文档收录了什么」这种没人会问的问题
    path = chunk["section_path"].strip()
    # 每篇末尾的「相关文件」是一串指向其他文档的链接，本身没有规则，而且列的
    # 全是别人的标题 —— 按关键词挑块时它命中最多，不排掉会把跨块样本全占走
    return bool(path) and not path.startswith("相关文件")


def sample_seeds(chunks: list[dict]) -> list[dict]:
    """每篇文档抽定额，层内保证表格块与短块各有一个。

    表格是原子块（一张表就是一个大块），短块在 BM25 里天然吃亏 —— 两类的检索
    行为都和普通正文块不同，靠随机抽碰运气会整批漏掉。
    """
    pool = [c for c in chunks if _usable(c)]
    lengths = sorted(len(c["body"]) for c in pool)
    short_line = lengths[len(lengths) // 3]

    rng = random.Random(SEED)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in pool:
        by_doc[chunk["doc_id"]].append(chunk)

    seeds: list[dict] = []
    for doc_id in sorted(by_doc):
        docs = by_doc[doc_id]
        quota = BASE_QUOTA + (1 if len(docs) >= BONUS_MIN_CHUNKS else 0)
        picked: list[dict] = []
        taken: set[str] = set()

        for group in (
            [c for c in docs if c["kind"] == "table"],
            [c for c in docs if len(c["body"]) <= short_line],
        ):
            rest = [c for c in group if c["chunk_id"] not in taken]
            if rest and len(picked) < quota:
                chosen = rng.choice(rest)
                picked.append(chosen)
                taken.add(chosen["chunk_id"])

        rest = [c for c in docs if c["chunk_id"] not in taken]
        rng.shuffle(rest)
        picked.extend(rest[: quota - len(picked)])
        seeds.extend(sorted(picked, key=lambda c: c["chunk_id"]))
    return seeds


def pick_pairs(chunks: list[dict]) -> list[tuple[str, dict, dict]]:
    """按关键词从两篇文档里各挑一个块，组成跨块样本的种子。"""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if _usable(chunk):
            by_doc[chunk["doc_id"]].append(chunk)

    def best(doc_id: str, keywords: str) -> dict:
        want = terms(keywords)

        def score(chunk: dict) -> tuple[int, str]:
            # 标题路径权重更高：它是这一条讲什么的直接标注，正文里同样的词多半
            # 只是顺带提到（P07 的「会员」在哪一条都有，只有 2.1 的标题是「无理由退货窗口」）。
            # chunk_id 参与排序做 tie-break：并列时挑同一个，重复跑结果不变
            hit = 3 * len(want & terms(chunk["section_path"])) + len(want & terms(chunk["body"]))
            return hit, chunk["chunk_id"]

        return max(by_doc[doc_id], key=score)

    pairs = []
    for topic, (doc_a, kw_a), (doc_b, kw_b) in MULTI_HOP:
        pairs.append((topic, best(doc_a, kw_a), best(doc_b, kw_b)))
    return pairs


# ── 生成 ──────────────────────────────────────────────────────────────────
def build_model():
    from llm import chat

    kwargs = {"temperature": 0, "timeout": 60, "max_retries": 2}
    # 结构化输出在兼容网关上的两个坑与改写那边一模一样（rag/retrieving/pipeline/rewrite.py）：
    # json_schema 常常没实现，走 function_calling；而开着 thinking 的模型不接受
    # function_calling 要设的 tool_choice，所以沿用同一个思考档位开关。
    effort = os.getenv("REFUND_AGENT_REWRITE_REASONING", "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    openai_side = chat.provider_of(chat.model_name("agent")) == chat.OPENAI
    method = "function_calling" if openai_side else ""
    return chat.build("agent", **kwargs).with_structured_output(
        Generated, **({"method": method} if method else {})
    )


def generate(model, prompt: str, body: str) -> Generated:
    """生成一组问题，口语档抄了原文就打回重写一次。"""

    def invoke(text: str) -> Generated:
        return model.invoke(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        )

    result = invoke(prompt)
    ratio = overlap_ratio(result.colloquial, body)
    if ratio <= MAX_COLLOQUIAL_OVERLAP:
        return result

    retried = invoke(prompt + RETRY.format(ratio=ratio, rejected=result.colloquial))
    # 重写反而更像原文就留着第一版，别让「重写」这一步自己变成噪声源
    return retried if overlap_ratio(retried.colloquial, body) < ratio else result


# ── 组装 ──────────────────────────────────────────────────────────────────
def _case(query: str, style: str, kind: str, seeds: list[dict], answer: str, meta: dict) -> dict:
    return {
        "case_id": "",  # 全部样本排完序后统一编号
        "query": query,
        "style": style,
        "type": kind,
        "seed_chunk_id": [c["chunk_id"] for c in seeds],
        "reference_answer": answer,
        "meta": meta,
    }


def _meta(seeds: list[dict], ratio: float, model_name: str) -> dict:
    return {
        "doc_id": "+".join(c["doc_id"] for c in seeds),
        "layer": "+".join(sorted({c["layer"] for c in seeds})),
        "kind": "+".join(sorted({c["kind"] for c in seeds})),
        "section": " / ".join(c["section_path"] or c["title"] for c in seeds),
        "overlap_ratio": ratio,
        "generated_by": model_name,
        "reviewed": False,
    }


def build_cases(model, model_name: str, seeds: list[dict], pairs: list) -> list[dict]:
    cases: list[dict] = []

    for i, chunk in enumerate(seeds, 1):
        print(f"  [{i}/{len(seeds)}] single {chunk['chunk_id']}", flush=True)
        got = generate(model, USER_SINGLE.format(body=chunk["body"]), chunk["body"])
        for style, query in (("formal", got.formal), ("colloquial", got.colloquial)):
            meta = _meta([chunk], overlap_ratio(query, chunk["body"]), model_name)
            cases.append(_case(query, style, "single", [chunk], got.reference_answer, meta))

    for i, (topic, a, b) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] multi_hop {a['chunk_id']} + {b['chunk_id']}", flush=True)
        both = a["body"] + "\n" + b["body"]
        got = generate(model, USER_MULTI.format(body_a=a["body"], body_b=b["body"]), both)
        for style, query in (("formal", got.formal), ("colloquial", got.colloquial)):
            meta = _meta([a, b], overlap_ratio(query, both), model_name)
            meta["topic"] = topic
            cases.append(
                _case(query, style, "multi_hop", [a, b], got.reference_answer, meta)
            )

    for topic, query in UNANSWERABLE:
        # 不调模型：这类 query 要的就是「语料里没有」，写死比让模型编更可靠
        cases.append(
            _case(
                query,
                "colloquial",
                "unanswerable",
                [],
                "",
                {
                    "doc_id": "",
                    "layer": "",
                    "kind": "",
                    "section": "",
                    "topic": topic,
                    "overlap_ratio": 0.0,
                    "generated_by": "handwritten",
                    "reviewed": True,
                },
            )
        )

    for n, case in enumerate(cases, 1):
        case["case_id"] = f"R1-{n:03d}"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="生成检索评测数据集 r1")
    parser.add_argument("--out", default=str(DATASET_DIR), help="输出目录")
    parser.add_argument("--limit", type=int, help="只生成前 N 个种子块（调提示词用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印抽样结果，不调模型")
    args = parser.parse_args()

    load_env()
    chunks = load_chunks()
    seeds = sample_seeds(chunks)
    pairs = pick_pairs(chunks)

    by_doc: dict[str, int] = defaultdict(int)
    for chunk in seeds:
        by_doc[chunk["doc_id"]] += 1
    print(f"语料 {len(chunks)} 块 → 种子块 {len(seeds)} 个 / 跨块对 {len(pairs)} 组")
    print("  分层：" + "  ".join(f"{d}×{n}" for d, n in sorted(by_doc.items())))
    print(
        f"  表格块 {sum(1 for c in seeds if c['kind'] == 'table')} 个，"
        f"法规层 {sum(1 for c in seeds if c['layer'] == 'law')} 个"
    )

    if args.limit:
        seeds, pairs = seeds[: args.limit], pairs[:1]
    if args.dry_run:
        for chunk in seeds:
            print(f"  {chunk['chunk_id']}  {chunk['section_path']}  {chunk['body'][:40]}…")
        return

    from llm import chat

    model_name = chat.model_name("agent")
    print(f"\n生成中（{model_name}，温度 0）：")
    cases = build_cases(build_model(), model_name, seeds, pairs)

    out = Path(args.out).resolve()
    write_cases(out, cases)
    print(f"\n{len(cases)} 条样本 → {out / 'cases.jsonl'}")
    print("下一步：python rag/evals/validate_cases.py")


if __name__ == "__main__":
    main()
