"""v1 装配 —— create_agent 把提示、工具、Context schema 组装成 Agent。"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from langchain.agents import create_agent

from agent.v1.prompt import SYSTEM_PROMPT
from agent.v1.tools import TOOLS
from app.context import RefundContext
from llm import chat

META = yaml.safe_load((Path(__file__).parent / "meta.yaml").read_text(encoding="utf-8"))

# 采样温度做成可配置项，而不是写死一个值。
#
# 退款决策属于「工具调用 / 合规」类任务，线上同样建议低温：同一个订单今天批、
# 明天拒是事故不是特性，投诉时要能重放、审计时要能解释。所以默认取 meta.yaml 里的 0。
#
# 留出环境变量是为了另一类评估：上线前的稳定性验收必须用**线上真实温度**跑多轮，
# 看方差和最坏情况 —— 在温度 0 下做的评估不代表线上表现。
#   离线回归（改动有没有让它变差）：默认低温、单轮，噪音最小、归因干净
#   稳定性验收（线上会不会翻车）：REFUND_AGENT_TEMPERATURE=<线上值>，连跑多轮
#
# 注意温度 0 不等于确定性输出：浮点非结合性、batch 组成、服务端负载均衡到不同
# 硬件或模型版本，都会带来差异。它是降噪，不是消噪。
TEMPERATURE = float(os.getenv("REFUND_AGENT_TEMPERATURE", str(META["temperature"])))

# meta.yaml 里的 model 是 anthropic 供应商下的默认值；切到 OPENAI_* 时它自动失效，
# 改由 OPENAI_MODEL 决定（规则见 llm/chat.py）。
MODEL_DEFAULT = META["model"]


@lru_cache(maxsize=1)
def build_agent():
    """装配并缓存 Agent。无状态，同一进程内复用同一个实例即可。"""
    return create_agent(
        model=chat.build("agent", MODEL_DEFAULT, temperature=TEMPERATURE),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # context_schema 是身份注入的入口：RefundContext 里的字段进得了工具层，
        # 进不了发给模型的 tool schema。
        context_schema=RefundContext,
    )
