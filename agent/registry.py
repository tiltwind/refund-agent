"""版本注册与选择（2-design 6.4）。

Agent 的每次改动都构成一个新版本，`agent/` 下按版本目录**并存**而非原地覆盖 ——
这样才能同集对比 v1/v2、定位退化用例、灰度回滚、按 agent_version 做线上归因。

v1 阶段只有一个版本，因此这里只有 get()。灰度路由（按流量比例 select）等
v2 落地后再加 —— 提前写一个只有一个候选的加权随机没有意义。
"""

from agent.v1 import graph as v1_graph

_VERSIONS = {
    "v1": v1_graph,
}


def get(version: str = "v1"):
    """按版本号取 Agent 实例。未知版本抛异常，不 fallback 到默认版本 ——
    拼写错误当场暴露，报告署的版本和实际跑的那一版始终一致。"""
    module = _VERSIONS.get(version)
    if module is None:
        raise ValueError(f"unknown agent version: {version}（可选：{list(_VERSIONS)}）")
    return module.build_agent()


def meta(version: str = "v1") -> dict:
    """取版本元信息，随 trace 上报。"""
    module = _VERSIONS.get(version)
    if module is None:
        raise ValueError(f"unknown agent version: {version}（可选：{list(_VERSIONS)}）")
    return module.META
