"""客户档案服务的接口与数据模型 —— 工具层只依赖这里，不依赖具体实现。"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class OrderBrief:
    """客户档案里附带的订单摘要（够模型核对订单归属即可，不含明细）。"""

    order_id: str
    product: str
    category: str
    price: float
    signed_days_ago: int
    """签收距今天数 —— 用相对天数而非绝对时间戳，见 2-design 3.3。"""
    refunded: bool


@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    level: str
    """普通会员 | 金牌会员 —— 影响无理由退货窗口。"""
    register_date: str
    refund_count_90d: int
    """近 90 天退款次数 —— 超阈值触发风控，关闭自动退款通道。"""
    orders: list[OrderBrief] = field(default_factory=list)


class CustomerService(Protocol):
    def get_profile(self, customer_id: str) -> CustomerProfile:
        """查询客户档案。归属校验由下游服务自己做（2-design 1.6）。"""
        ...
