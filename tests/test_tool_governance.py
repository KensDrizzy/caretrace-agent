import unittest
from types import SimpleNamespace

from app.core.enums import RiskLevel, ToolJobKind
from app.services.tool_governance import ToolPolicyRegistry


def report(risk: RiskLevel):
    return SimpleNamespace(risk_level=risk.value)


class ToolGovernanceTests(unittest.TestCase):
    def test_high_risk_alert_is_allowed_only_for_high_risk(self):
        allowed, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.ALERT_SEND.value, report(RiskLevel.HIGH))
        blocked, reason, _ = ToolPolicyRegistry.authorize(ToolJobKind.ALERT_SEND.value, report(RiskLevel.LOW))

        self.assertTrue(allowed)
        self.assertFalse(blocked)
        self.assertIn("不允许", reason)

    def test_medium_case_create_is_allowed_but_low_is_blocked(self):
        allowed, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.CASE_CREATE.value, report(RiskLevel.MEDIUM))
        blocked, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.CASE_CREATE.value, report(RiskLevel.LOW))

        self.assertTrue(allowed)
        self.assertFalse(blocked)

    def test_unknown_tool_is_blocked(self):
        allowed, reason, policy = ToolPolicyRegistry.authorize("DELETE_EVERYTHING", report(RiskLevel.HIGH))

        self.assertFalse(allowed)
        self.assertIsNone(policy)
        self.assertIn("未知工具", reason)


if __name__ == "__main__":
    unittest.main()
