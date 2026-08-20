import os
import tempfile
import unittest
from unittest.mock import patch

from services import prod_store
from services.customer.prod import ProdCustomerService
from services.order.prod import ProdOrderService
from services.rule.prod import ProdRuleService


class ProdServicesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {"REFUND_AGENT_SQLITE_PATH": f"{self.temp_dir.name}/sqlite.db"},
        )
        self.env.start()
        prod_store.initialize()

    def tearDown(self):
        self.env.stop()
        self.temp_dir.cleanup()

    def test_customer_profile_reads_orders(self):
        profile = ProdCustomerService().get_profile("C1001")

        self.assertEqual(profile.level, "金牌会员")
        self.assertEqual([order.order_id for order in profile.orders], ["O2001"])

    def test_rule_service_reads_customer_and_order(self):
        service = ProdRuleService()

        approved = service.check_eligibility("O2001", "C1001", "无理由", "未拆封")
        category_denial = service.check_eligibility("O2002", "C1002", "质量问题")
        risk_denial = service.check_eligibility("O2003", "C1003", "质量问题")
        clarify = service.check_eligibility("O2004", "C1004")

        self.assertEqual((approved.verdict, approved.refundable_amount), ("通过", 899.0))
        self.assertEqual(category_denial.verdict, "不通过")
        self.assertEqual(risk_denial.verdict, "不通过")
        self.assertEqual(clarify.verdict, "需补充")

    def test_refund_is_persisted_and_idempotent(self):
        service = ProdOrderService()

        first = service.execute_refund("O2001", "C1001", 899.0, "符合规则", "req-1")
        replay = service.execute_refund("O2001", "C1001", 899.0, "符合规则", "req-1")

        self.assertEqual(first, replay)
        self.assertEqual(len(prod_store.decision_log()), 1)
        with prod_store.connect() as db:
            refunded = db.execute(
                "SELECT refunded FROM orders WHERE order_id = 'O2001'"
            ).fetchone()[0]
        self.assertEqual(refunded, 1)

    def test_denial_is_persisted(self):
        receipt = ProdOrderService().record_denial(
            "O9999", "C1001", "订单不存在", "req-2"
        )

        self.assertTrue(receipt.receipt_no.startswith("D"))
        self.assertEqual(prod_store.decision_log()[0]["decision"], "拒绝")


if __name__ == "__main__":
    unittest.main()
