"""客户档案服务的 prod 实现 —— 待接入用户服务。

v1 只跑通 eval 数据源，prod 实现留桩。接入时在这里发 HTTP：

    GET {USER_SVC}/customers/me
    Authorization: Bearer <agent 服务自己的 service token>
    X-Acting-User: {customer_id}      ← 不透传用户 JWT（2-design 1.5）
    X-Request-Id / traceparent

归属校验由用户服务自己做，Agent 只如实转述身份。公共 header 的注入、重试、
熔断、超时统一放 services/base.py，这里只写业务映射。
"""

from services.customer.protocol import CustomerProfile


class ProdCustomerService:
    def get_profile(self, customer_id: str) -> CustomerProfile:
        raise NotImplementedError("prod 客户档案服务未接入：v1 仅支持 request_source=eval")
