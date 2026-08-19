"""r1 数据集脚本的公共部分：环境变量、语料、jsonl、judge 模型。

语料从 **Milvus 取**，不从 `doc/policy/` 重新切一遍。数据集绑的是 `chunk_id`，
而 `chunk_id` 由切片位置派生（rag/chunking/policy.py），脚本自己再切一次就等于
在库外维护第二份切片产物：切分参数一改，两边悄悄错开，生成时看着对、评测时全线
判负。collection 是唯一事实源，生成与自检读的都是它。
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "rag" / "datasets" / "r1"

# 比对文本时忽略的虚词。只在两个字都是虚词时才丢掉这个二元组 ——
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


# ── 关键词匹配 ────────────────────────────────────────────────────────────
def terms(text: str) -> set[str]:
    """把文本切成可比对的实词单位：中文取相邻二元组，数字与拉丁词整体取。

    这里**没有分词**：项目不装 jieba，BM25 的中文分析器跑在 Milvus 服务端
    （rag/index/seed_milvus.py），本机拿不到同一套切词。二元组是零依赖下最接近
    中文词的近似 —— 「无理由退货」切出「无理」「理由」「由退」「退货」。

    它只用来给跨块样本按关键词挑块（`generate_cases.py`），不参与判分，
    近似带来的误差可以接受。
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


# ── judge 模型 ────────────────────────────────────────────────────────────
JUDGE_DEFAULT = "anthropic:claude-sonnet-5"
"""judge 的兜底模型，只在 provider=anthropic 时生效（规则见 llm/chat.py）。

不跟改写共用便宜档：改写判错顶多让排序掉一档，judge 判错直接把指标变成噪声。
走 OPENAI_* 时配 OPENAI_JUDGE_MODEL —— 兼容网关的模型名猜不出来。
"""

def judge_structured() -> str:
    """结构化输出的实现方式，留空则 OpenAI 侧走 function_calling、Anthropic 侧走默认。

    与改写那边同一个坑：langchain 对 OpenAI 侧默认 `json_schema`（strict mode），
    DeepSeek 这类网关回 400。可选 json_schema | function_calling | json_mode。

    这几个开关**读的时候才取环境变量**，不写成模块常量：评估脚本是直接 python 跑的，
    `.env` 由 `load_env()` 在 main 里补进 os.environ，那时 import 早就发生过了。
    """
    return os.getenv("REFUND_AGENT_JUDGE_STRUCTURED", "").strip()


def judge_reasoning() -> str:
    """思考档位，留空不传。

    开着 thinking 的模型往往不接受 function_calling 需要的 `tool_choice`
    （DeepSeek 回 400），填 `none` 关掉。非推理模型收到这个参数会报错，所以不设默认值。
    """
    return os.getenv("REFUND_AGENT_JUDGE_REASONING", "").strip()


@lru_cache(maxsize=1)
def judge_name() -> str:
    """judge 用哪个模型。同时检查它有没有跟链路的改写模型撞车。

    撞车了指标仍然算得出来，但那是同一个模型给自己的检索结果打分，方向性偏差
    没法从分数上看出来 —— 所以在跑批一开始就说清楚，而不是等报告出来再解释。
    """
    from llm import chat
    from rag.retrieving.pipeline.rewrite import MODEL_DEFAULT as REWRITE_DEFAULT

    name = chat.model_name("judge", JUDGE_DEFAULT)
    if name == chat.model_name("rewrite", REWRITE_DEFAULT):
        print(
            f"[warn] judge 与链路的改写模型同为 {name}：自己评自己。"
            "配 OPENAI_JUDGE_MODEL / ANTHROPIC_JUDGE_MODEL 分开。"
        )
    return name


def judge_json_hint(schema) -> str:
    """json_mode 下要追加到提示词末尾的一段：字段结构。

    `json_mode` 是 DeepSeek 这类网关上**唯一能同时开着思考模式**的结构化方式
    （function_calling 要设 tool_choice，与 thinking 冲突），而逐条判定正是该让
    模型多想一步的活。代价是它只保证「输出是合法 JSON」：langchain 不会像
    function_calling 那样把 schema 传给模型，字段名得自己写进提示词。实测里模型给过
    `{"useful": [...]}` 这样的自造字段名，解析直接失败。

    schema 由 pydantic 模型现生成，不手写 —— 手写的那份迟早跟类定义对不上。
    """
    if judge_structured() != "json_mode":
        return ""
    fields = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return f"\n\n以 JSON 对象输出结果，且必须符合这个 JSON Schema：\n{fields}"


def build_judge(schema, **kwargs):
    """构造吐结构化结果的 judge 模型。温度 0 是硬要求：判定要可复现。"""
    from llm import chat

    name = judge_name()
    kwargs.setdefault("temperature", 0)
    if effort := judge_reasoning():
        kwargs["reasoning_effort"] = effort

    method = judge_structured() or (
        "function_calling" if chat.provider_of(name) == chat.OPENAI else ""
    )
    model = chat.build("judge", JUDGE_DEFAULT, **kwargs)
    return model.with_structured_output(schema, **({"method": method} if method else {}))
