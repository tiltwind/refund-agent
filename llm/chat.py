"""对话模型接入 —— 供应商与模型名的唯一解析处。

项目里有三处要调对话模型：Agent 主循环（agent/v1/graph.py）、检索链路的查询改写
（rag/retrieving/pipeline/rewrite.py）、评测的 LLM judge（rag/experiments/*/judge.py）。
各自 `os.getenv` 拼模型名，很快就会漂移成「主模型换了供应商、改写还在打另一家的
端点」——所以解析只做一次，放这里。

## 选哪家

    REFUND_AGENT_PROVIDER=anthropic|openai     # 显式指定，最高优先级
    没指定时：哪边的 API key 非空就走哪边；两边都配了，anthropic 优先

「两边都配了走 anthropic」不是偏好，是**不改变既有行为**：这个项目原本只有
ANTHROPIC_*，把 OPENAI_* 加进来后，老的 .env 必须跑出跟以前一样的结果。要切到
另一边就显式写 REFUND_AGENT_PROVIDER —— 让切换是一个动作，而不是删掉某个 key
的副作用。

## 用哪个模型（优先级从高到低）

    1. REFUND_AGENT_MODEL / REFUND_AGENT_REWRITE_MODEL / REFUND_AGENT_JUDGE_MODEL   跨供应商的强制覆盖
    2. OPENAI_MODEL / ANTHROPIC_MODEL（按当前 provider 取）
    3. 调用方传的 default（仅当它属于当前 provider）
    4. 改写 / judge 角色回落到主模型；主模型没有兜底值时报错

第 3 条的限定很关键：agent/v1/meta.yaml 里写的是 `anthropic:claude-sonnet-5`，
provider 切到 openai 后这个默认值必须失效，否则会拿着 Claude 的模型名去打
OpenAI 的端点，报一个跟真实原因（没配 OPENAI_MODEL）毫不相干的 404。

第 4 条对 openai **不设内置默认值**：走 OPENAI_* 的多半是兼容网关（DeepSeek、
Qwen、vLLM…），模型名各家各写各的，猜任何一个都是错的。与其让它 404，不如直接
告诉用户配 OPENAI_MODEL。

## 打哪个端点、用哪把 key

端点与凭据跟模型名同一套角色前缀，角色专属的优先，没配就回落到通用的：

    OPENAI_JUDGE_BASE_URL / OPENAI_JUDGE_API_KEY   → OPENAI_BASE_URL / OPENAI_API_KEY
    ANTHROPIC_REWRITE_BASE_URL / …_API_KEY         → ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY

只换模型名、端点照旧是最常见的配法，所以回落是必须的。反过来 judge 换到另一个
网关也很常见 —— 它该比主模型强一档，而那一档未必在同一个端点上，换端点通常就得
换 key。凭据不做回落：没单独配就不传，让 SDK 自己读通用的那把（见 api_key）。
"""

import os
from functools import lru_cache

ANTHROPIC = "anthropic"
OPENAI = "openai"
PROVIDERS = (ANTHROPIC, OPENAI)

# 每个角色两级环境变量：通用覆盖 + 供应商专属。
_ROLE_ENVS = {
    "agent": ("REFUND_AGENT_MODEL", {ANTHROPIC: "ANTHROPIC_MODEL", OPENAI: "OPENAI_MODEL"}),
    "rewrite": (
        "REFUND_AGENT_REWRITE_MODEL",
        {ANTHROPIC: "ANTHROPIC_REWRITE_MODEL", OPENAI: "OPENAI_REWRITE_MODEL"},
    ),
    # judge 只在评测脚本里用，不在线上链路上。它该配一个比主模型更强的档次：判定
    # 「这条 claim 有没有被上下文支撑」比生成一句改写难，judge 判错，指标就是噪声。
    "judge": (
        "REFUND_AGENT_JUDGE_MODEL",
        {ANTHROPIC: "ANTHROPIC_JUDGE_MODEL", OPENAI: "OPENAI_JUDGE_MODEL"},
    ),
}


# 端点与凭据的角色前缀。模型名有 _ROLE_ENVS 那套两级变量，端点和 key 也得跟着分开：
# judge 该比主模型强一档，而那一档常常在另一个网关、另一个账号、另一档配额上。
_ROLE_PREFIX = {"agent": "", "rewrite": "REWRITE_", "judge": "JUDGE_"}


def _env(name: str) -> str:
    """读环境变量并去空白 —— .env 里写成 KEY="" 或带尾随空格是常态。"""
    return os.getenv(name, "").strip()


def _role_env(vendor: str, role: str, *names: str) -> str:
    """角色专属变量优先，没配就回落到通用的那个：

        OPENAI_JUDGE_BASE_URL → OPENAI_BASE_URL → OPENAI_API_BASE

    回落是必须的：只想换 judge 的模型、端点照旧，是最常见的配法。
    """
    prefix = _ROLE_PREFIX[role]
    heads = [f"{vendor.upper()}_{prefix}", f"{vendor.upper()}_"] if prefix else [f"{vendor.upper()}_"]
    for head in heads:
        for name in names:
            if value := _env(head + name):
                return value
    return ""


@lru_cache(maxsize=1)
def provider() -> str:
    explicit = _env("REFUND_AGENT_PROVIDER").lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(f"未知的 REFUND_AGENT_PROVIDER：{explicit}（可选：{list(PROVIDERS)}）")
        return explicit

    if _env("ANTHROPIC_API_KEY"):
        return ANTHROPIC
    if _env("OPENAI_API_KEY"):
        return OPENAI

    raise RuntimeError(
        "没有可用的模型凭据：ANTHROPIC_API_KEY 与 OPENAI_API_KEY 都为空。\n"
        "在 .env 里配置其中之一（参考 .env.example），或用 REFUND_AGENT_PROVIDER 指定供应商。"
    )


def available() -> bool:
    """有没有可用凭据。查询改写这类可降级的调用方用它先探一下，避免拿异常当控制流。"""
    try:
        provider()
    except (RuntimeError, ValueError):
        return False
    return True


def _qualify(name: str, current: str) -> str:
    """补全 `provider:model` 前缀。

    只把已知供应商名当前缀，不是见到冒号就当前缀 —— 有些网关的模型名自带冒号
    （`qwen2.5:7b`、`llama3:latest`），误判会拼出 `qwen2.5:openai:7b` 这种东西。
    """
    head, _, rest = name.partition(":")
    if rest and head in PROVIDERS:
        return name
    return f"{current}:{name}"


def model_name(role: str = "agent", default: str = "") -> str:
    """解析出 init_chat_model 认识的 `provider:model` 串。"""
    if role not in _ROLE_ENVS:
        raise ValueError(f"未知的模型角色：{role}（可选：{list(_ROLE_ENVS)}）")

    current = provider()
    generic_env, per_provider_env = _ROLE_ENVS[role]

    for candidate in (_env(generic_env), _env(per_provider_env[current])):
        if candidate:
            return _qualify(candidate, current)

    # 调用方给的默认值只在同供应商下有效（见模块 docstring 第 3 条）
    if default:
        qualified = _qualify(default, current)
        if qualified.startswith(f"{current}:"):
            return qualified

    if role != "agent":
        # 没单独配就跟着主模型走：兼容网关那边没有「便宜档」「更强档」的固定名字可猜。
        # judge 落到这条路径意味着自己评自己，评测侧会为此打一行警告（rag judge.py）。
        return model_name("agent")

    if current == ANTHROPIC:
        return f"{ANTHROPIC}:claude-sonnet-5"
    raise RuntimeError(
        "provider=openai 但没有指定模型名：请配置 OPENAI_MODEL（或 REFUND_AGENT_MODEL）。\n"
        "兼容网关的模型名各家不同，这里不猜默认值。"
    )


def provider_of(model: str) -> str:
    """这个模型串属于哪家。

    通常就是当前 provider()，但 REFUND_AGENT_MODEL 允许带前缀跨供应商覆盖
    （`REFUND_AGENT_MODEL=openai:qwen-max` 配 ANTHROPIC_API_KEY 的环境）——
    那种情况下端点必须跟着模型走，否则会拿 OpenAI 的模型名去打 Anthropic 的地址。
    """
    head, _, rest = model.partition(":")
    return head if rest and head in PROVIDERS else provider()


def base_url(target: str = "", role: str = "agent") -> str:
    """某家供应商的端点，空串表示走官方默认。target 省略时取当前 provider。"""
    if (target or provider()) == OPENAI:
        # OPENAI_BASE_URL 是 openai SDK 的变量名，而 langchain-openai 读的是
        # OPENAI_API_BASE —— 两个都认，取到值后显式传参，不依赖谁读谁。
        return _role_env(OPENAI, role, "BASE_URL", "API_BASE")
    return _role_env(ANTHROPIC, role, "BASE_URL")


def api_key(role: str = "agent", target: str = "") -> str:
    """角色专属凭据。**没单独配就返回空串**，让 SDK 自己去读通用的那把 key。

    不在这里回落到通用 key：回落等于把凭据从 SDK 手里接过来再传回去，多一处
    覆盖点却不多解决一个问题。单独配了才显式传 —— 换 judge 的端点通常就得换 key，
    两个网关不会认同一把。
    """
    vendor = OPENAI if (target or provider()) == OPENAI else ANTHROPIC
    prefix = _ROLE_PREFIX[role]
    return _env(f"{vendor.upper()}_{prefix}API_KEY") if prefix else ""


def build(role: str = "agent", default: str = "", **kwargs):
    """构造对话模型实例。kwargs 原样透传给 init_chat_model（temperature、timeout…）。"""
    from langchain.chat_models import init_chat_model

    name = model_name(role, default)
    target = provider_of(name)
    # Anthropic 侧不传 base_url：ChatAnthropic 自己读 ANTHROPIC_BASE_URL，传了反而
    # 多一处覆盖点。OpenAI 侧必须显式传，因为变量名对不上（见 base_url 注释）。
    if target == OPENAI and (url := base_url(OPENAI, role)):
        kwargs.setdefault("base_url", url)
    if key := api_key(role, target):
        kwargs.setdefault("api_key", key)
    return init_chat_model(name, **kwargs)


def describe() -> str:
    """一行状态摘要，给启动日志用 —— 「到底在打哪个端点的哪个模型」要一眼可见。"""
    if not available():
        return "模型: 未配置（ANTHROPIC_API_KEY / OPENAI_API_KEY 都为空）"
    try:
        agent_model = model_name("agent")
    except RuntimeError as exc:
        return f"模型: provider={provider()} 但解析失败（{exc.args[0].splitlines()[0]}）"

    rewrite_model = model_name("rewrite", "anthropic:claude-haiku-4-5")
    endpoint = base_url(provider_of(agent_model)) or "官方端点"
    same = "同上" if rewrite_model == agent_model else rewrite_model
    return f"模型: {agent_model} → {endpoint}（改写: {same}）"
