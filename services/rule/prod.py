"""规则服务的 prod 实现 —— 待接入。

v1 只跑通 eval 数据源，prod 实现留桩。接入时在这里发 HTTP：

    POST {RULE_SVC}/refund-eligibility     X-Acting-User: {acting_user}

**规则引擎不要搬到这一侧来。** 规则口径由规则服务团队独立发版，Agent 只消费
判定结论；把规则抄进 Agent 服务，等于每次改窗口天数都要发一次 Agent
（1-architecture 第一章）。这里只做协议映射。
"""

from services.rule.protocol import EligibilityResult

_UNAVAILABLE = "prod 规则服务未接入：v1 仅支持 request_source=eval"


class ProdRuleService:
    def check_eligibility(
        self,
        order_id: str,
        acting_user: str,
        reason_type: str = "",
        item_condition: str = "",
    ) -> EligibilityResult:
        raise NotImplementedError(_UNAVAILABLE)
