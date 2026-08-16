"""Telemetry —— Agent 链路上报到 Langfuse（2-design 第五章）。

Langfuse v3 起底座换成了 OpenTelemetry：CallbackHandler 把 LangGraph 的图节点、
工具调用、LLM generation 转成 OTel span 后经 OTLP 上报。所以这里不需要自己搭
tracer provider，接一个 callback 就能拿到 8.1 那棵调用树。

三条设计约束：

1. **缺密钥就静默降级**。埋点是旁路，不该让主链路挂掉 —— 本地跑 demo、CI 里跑
   评估都可能没有 Langfuse。这时 trace_config() 返回一个不带 callbacks 的普通
   config，调用方不必写 if。

2. **脱敏做在 mask 钩子里，不做在调用点**（2-design 5.4）。mask 是 SDK 级钩子，
   所有 span 的 input/output 都要过它一遍，而不是指望每个埋点自己记得脱敏。线上
   评估的 LLM judge 读的是同一批 trace，漏一处 PII 就跟着进了 judge 的 prompt。

3. **customer_id 上报哈希而非原值**（2-design 5.2）。要的是「同一个人的多次请求能
   串起来」，不是「知道他是谁」—— 加盐哈希两者都满足。
"""

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from app.context import RefundContext

# 兜底脱敏规则。顺序有讲究：18 位身份证会被银行卡的 16-19 位数字规则吃掉，必须排在前面。
#
# 收货地址和真实姓名靠正则认不准（"北京市"和"张伟"都是普通中文串），真正的做法是
# 在写入 span 属性的地方就别把它们带进来 —— 下面这几条是最后一道网，不是全部防线。
_PII_RULES = [
    (re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"), "<id-card>"),
    (re.compile(r"\b\d{16,19}\b"), "<bank-card>"),
    (re.compile(r"1[3-9]\d{9}"), "<phone>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
]

# 哈希加盐。同一套盐下 customer_id → user_id 的映射稳定，换盐会让新旧 trace 对不上，
# 所以它属于配置而非随机值；线上从密钥管理里取，别用这个默认值。
_HASH_SALT = os.getenv("REFUND_AGENT_HASH_SALT", "refund-agent")


def mask(*, data: Any, **_: Any) -> Any:
    """SDK 的 MaskFunction：递归遍历 span 的 input/output，把 PII 替换成占位符。"""
    if isinstance(data, str):
        for pattern, placeholder in _PII_RULES:
            data = pattern.sub(placeholder, data)
        return data
    if isinstance(data, dict):
        return {key: mask(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(data=item) for item in data]
    return data


def hash_customer(customer_id: str) -> str:
    """把 customer_id 转成脱敏但稳定的 user_id（2-design 5.2）。"""
    return hashlib.sha256(f"{_HASH_SALT}:{customer_id}".encode()).hexdigest()[:16]


def base_url() -> str:
    """两个变量名都认。

    v4 SDK 同时读 LANGFUSE_BASE_URL 和 LANGFUSE_HOST，v2/v3 只认 LANGFUSE_HOST ——
    这里显式取值再显式传参，换 SDK 版本时不会因为变量名不被识别，就悄悄地把数据
    发去 cloud.langfuse.com（默认值），而本地那个 Langfuse 上一条 trace 都没有。
    """
    return os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or ""


@lru_cache(maxsize=1)
def _handler():
    """构造并缓存 CallbackHandler；未配置或未安装 SDK 时返回 None。

    Langfuse v3+ 的 CallbackHandler 自己不接凭据，它从进程内的全局客户端单例取 ——
    所以要先 Langfuse(...) 初始化一次，顺序反了会拿到一个被禁用的 handler。
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not (public_key and secret_key):
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:  # 没装 langfuse 也能跑主链路，只是没有 trace
        return None

    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=base_url() or None,
        mask=mask,
    )
    return CallbackHandler()


def enabled() -> bool:
    return _handler() is not None


def trace_config(ctx: RefundContext, meta: dict | None = None, *, name: str = "refund-chat") -> dict:
    """构造 invoke 用的 config —— 一次请求一条 trace（2-design 5.1）。

    meta 走参数而不是在这里 import agent.registry：services 是被 agent 依赖的下层，
    反向 import 会成环。调用方本来就拿得到 registry.meta(version)。
    """
    handler = _handler()
    if handler is None:
        return {}

    meta = meta or {}
    # langfuse_* 是 CallbackHandler 约定的保留键，会被提到 trace 层；其余键原样落在
    # metadata 里，供 Langfuse 上按版本/来源筛选。
    metadata = {
        "langfuse_session_id": ctx.session_id or ctx.request_id,
        "langfuse_user_id": hash_customer(ctx.customer_id),
        "langfuse_tags": [
            f"source:{ctx.request_source}",
            f"agent:{meta.get('agent_version', 'unknown')}",
            f"prompt:{meta.get('prompt_version', 'unknown')}",
        ],
        # 评估归因的关键：线上指标掉了，要能立刻回答「是哪次发版引起的」（2-design 5.2）
        "agent_version": meta.get("agent_version"),
        "prompt_version": meta.get("prompt_version"),
        "request_id": ctx.request_id,
        "request_source": ctx.request_source,
        "actor": ctx.actor,
    }
    return {"callbacks": [handler], "metadata": metadata, "run_name": name}


def flush() -> None:
    """进程退出前把缓冲区里的 span 推出去。

    SDK 是攒批异步上报的（flush_at / flush_interval），短命脚本跑完就退出，还没到
    发送时机的那批 span 会随进程一起消失 ——「跑完了但 Langfuse 上什么都没有」多半
    是漏了这一步，而不是没接上。长驻服务（app/main.py）不用每次请求调，在 lifespan
    的 shutdown 里调一次即可。
    """
    if _handler() is None:
        return
    from langfuse import get_client

    get_client().flush()


def describe() -> str:
    """一行状态摘要，给启动日志用。

    顺手做一次 auth_check：凭据写错、Langfuse 服务没起这类问题，在这里报出来只要
    一秒，等跑完三个场景才发现没 trace 就晚了。它只在启动时调一次，不在请求路径上。
    """
    if not enabled():
        return "Langfuse: off（未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，链路照常跑，只是没有 trace）"

    target = base_url() or "https://cloud.langfuse.com（默认）"
    try:
        from langfuse import get_client

        ok = get_client().auth_check()
    except Exception as exc:  # 网络不通、SDK 版本没有 auth_check —— 都不该拦住主链路
        return f"Langfuse: on → {target}（凭据校验未通过：{exc}）"
    return f"Langfuse: {'on' if ok else 'on（凭据校验失败，trace 不会入库）'} → {target}"
