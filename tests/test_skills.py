import tempfile
import unittest
from pathlib import Path

from app.services.skills import MindBridgeSkillRegistry, SkillLoadError


def write_skill(root: Path, name: str, text: str) -> None:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


class SkillRegistryTests(unittest.TestCase):
    def test_skill_registry_loads_valid_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(
                root,
                "demo_skill",
                """---\nname: demo_skill\ndescription: Use for a clear and sufficiently described demo scenario.\n---\n\n# Demo\n\n## Workflow\n\n- Do one thing.\n""",
            )

            skill = MindBridgeSkillRegistry(root).get_required("demo_skill")

            self.assertEqual(skill.name, "demo_skill")
            self.assertEqual(skill.validation_issues(), [])

    def test_skill_status_reports_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(
                root,
                "demo_skill",
                """---\nname: demo_skill\ndescription: short\n---\n\n# Demo\n""",
            )

            status = MindBridgeSkillRegistry(root).status_items()[0]

            self.assertEqual(status["status"], "WARN")
            self.assertTrue(status["issues"])

    def test_skill_requires_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(root, "bad", "# Missing metadata")

            with self.assertRaises(SkillLoadError):
                MindBridgeSkillRegistry(root).get_required("bad")


if __name__ == "__main__":
    unittest.main()


def test_selected_skill_does_not_load_unrelated_broken_body(tmp_path):
    write_skill(tmp_path, 'good', '---\nname: good\ndescription: A sufficiently detailed valid skill description.\n---\n## Workflow\nDo one thing.')
    write_skill(tmp_path, 'broken', '# invalid unrelated metadata')
    assert MindBridgeSkillRegistry(tmp_path).get_required('good').name == 'good'


def test_high_risk_skill_selection_takes_priority_over_chat_intent():
    from app.core.enums import IntentType, RiskLevel
    from app.services.skills import MindBridgeSkillLibrary
    assert MindBridgeSkillLibrary.response_skill_names(IntentType.CHAT, RiskLevel.HIGH, '你好') == [
        'supportive_response_baseline', 'high_risk_safety_plan']
    assert MindBridgeSkillLibrary.response_skill_names(IntentType.CHAT, RiskLevel.LOW, '你好') == []
    assert 'sleep_routine_support' in MindBridgeSkillLibrary.response_skill_names(
        IntentType.CONSULT, RiskLevel.LOW, '最近失眠')


def test_selected_skill_rejects_path_traversal(tmp_path):
    import pytest
    with pytest.raises(SkillLoadError):
        MindBridgeSkillRegistry(tmp_path).get_required('../secret')
