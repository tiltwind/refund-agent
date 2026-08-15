"""按 request_source 选实现 —— 工具层唯一的入口（README 6.2）。

工具层拿到的永远是 protocol 里的接口，不知道也不关心数据从哪来。
未来加 replay 数据源，只需在这里多一个 case 分支。
"""

from app.context import RefundContext
from services.customer.eval import EvalCustomerService
from services.customer.prod import ProdCustomerService
from services.customer.protocol import CustomerService
from services.order.eval import EvalOrderService
from services.order.prod import ProdOrderService
from services.order.protocol import OrderService
from services.rag.milvus import MilvusRagService
from services.rag.protocol import RagService

# 未知的 request_source 一律抛异常，**绝不 fallback 到 prod** ——
# 拼错一个字母就静默连上线上库，是这类工厂函数最典型的事故（README 6.2）。
_UNKNOWN = "unknown request_source: {}"


def customer_service(ctx: RefundContext) -> CustomerService:
    match ctx.request_source:
        case "prod":
            return ProdCustomerService()
        case "eval":
            return EvalCustomerService()
        case other:
            raise ValueError(_UNKNOWN.format(other))


def order_service(ctx: RefundContext) -> OrderService:
    match ctx.request_source:
        case "prod":
            return ProdOrderService()
        case "eval":
            return EvalOrderService()
        case other:
            raise ValueError(_UNKNOWN.format(other))


def rag_service(ctx: RefundContext) -> RagService:
    """政策检索**不按 request_source 分实现**：prod 与 eval 都直连 Milvus，
    走同一个 collection、同一条检索路径（README 6.4）。

    ctx 仍然传进来，是为了后续把服务身份、traceparent、超时这些横切关注点
    接上去 —— 接口保持一致，将来真要换实现也不必改调用方。"""
    return MilvusRagService()
