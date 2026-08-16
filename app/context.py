"""RefundContext —— 一次请求的运行时上下文（README 4.3）。

这个对象是「身份来自认证、不来自对话」这条原则的落地点：
customer_id / actor 由网关从 JWT claims 提取后注入 header，认证中间件读出来
构造 RefundContext，再由 create_agent 的 context_schema 传给工具层。

它**不会**出现在发给模型的 tool schema 里 —— 模型看不见、也改不了自己
正在操作谁的数据。把 customer_id 放进工具参数等于把「访问谁的数据」的决策权
交给一个可被 prompt injection 操控的组件（IDOR 越权）。
"""

from dataclasses import dataclass


@dataclass
class RefundContext:
    customer_id: str
    """主体身份，由网关从 JWT claims.sub 提取后注入。模型不可见、不可改。"""

    actor: str = "self"
    """操作者：self | staff:{staff_id}。审计流水要用，缺了事后追不到人。"""

    request_id: str = ""
    """贯穿全链路的追踪 ID，兼作退款执行的幂等键（README 第七章）。"""

    session_id: str = ""
    """同一通会话的多轮请求共用的 ID，随 trace 上报（README 8.2）。

    和 request_id 是两个粒度：一次请求一个 request_id，一通会话一个 session_id。
    排障时经常要看「用户在这轮之前问了什么」，只有 request_id 串不起来。
    留空时 telemetry 回落到 request_id，trace 至少不会互相污染。
    """

    request_source: str = "prod"
    """prod | eval —— 决定 services/ 选哪个实现。

    必须由服务端决定，绝不能由客户端请求携带：否则任何调用方声明一句
    request_source=eval 就能绕开真实数据与真实风控（README 9.3）。
    """
