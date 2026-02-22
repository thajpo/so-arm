from pathlib import Path
import unittest


class RepoContractTests(unittest.TestCase):
    def test_current_md_has_required_sections(self) -> None:
        required_sections = (
            "## Institutional Knowledge",
            "## Beliefs",
            "## Brainstormed",
            "## Specd",
        )
        text = Path("current.md").read_text(encoding="utf-8")
        for section in required_sections:
            self.assertIn(section, text, msg=f"Missing required section: {section}")

    def test_github_workflow_and_templates_exist(self) -> None:
        required_paths = (
            ".github/ISSUE_TEMPLATE/spec-contract.md",
            ".github/pull_request_template.md",
            ".github/workflows/pr-checks.yml",
        )
        for relative_path in required_paths:
            self.assertTrue(
                Path(relative_path).exists(),
                msg=f"Missing required file: {relative_path}",
            )


if __name__ == "__main__":
    unittest.main()
