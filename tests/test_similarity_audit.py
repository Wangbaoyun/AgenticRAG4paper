#!/usr/bin/env python3
"""``tools/similarity_audit.py`` 的单元测试（仅使用标准库 unittest，无需 pytest）。

运行::

    python3 -m unittest discover -s tests -p "test_similarity_audit.py" -v
    # 或
    python3 tests/test_similarity_audit.py

覆盖范围
--------
* 完全无重叠的两棵迷你目录树 → 全部检查通过、退出码 0；
* 注入一段从参考树复制的 token 序列 → 检查 1（n-gram）失败；
* 注入高度相似的字符串字面量 → 检查 3（Prompt 泄露）失败；
* pyproject.toml 写 ``paper-qa`` → 检查 4（依赖）失败；
* 专有标识符出现在代码 vs 出现在注释 → 分别判违规 / 警告；
* 搬运上游资产（内容哈希相同）→ 检查 5 失败；
* 临时 git 仓库验证「未跟踪文件被排除、工作区改动被忽略」；
* 参数 / 路径错误 → 退出码 2；报告父目录自动创建。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "similarity_audit.py"


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("similarity_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["similarity_audit"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


# --------------------------------------------------------------------------- #
# 测试辅助
# --------------------------------------------------------------------------- #

#: 参考树里的原创代码（测试专用，绝不来自真实上游项目）。
REFERENCE_SOURCE = '''
"""Reference sample module used only by the unit tests."""

import math


class OrbitCalculator:
    """Compute elliptical orbit periods for the audit fixture."""

    def __init__(self, semi_major_axis: float) -> None:
        self.semi_major_axis = semi_major_axis

    def period(self, central_mass: float) -> float:
        numerator = 2.0 * math.pi * (self.semi_major_axis ** 1.5)
        denominator = math.sqrt(6.674e-11 * central_mass)
        return numerator / denominator
'''

#: 目标树里的原创代码：词汇与结构都与参考树不同。
TARGET_SOURCE = '''
"""Independent module: bread recipes."""

from dataclasses import dataclass


@dataclass
class SourdoughRecipe:
    hydration_percent: int
    bulk_hours: int

    def describe(self) -> str:
        share = self.hydration_percent / 100
        return "hydration share " + str(round(share, 2))
'''

#: 一段被"搬运"的目标代码：token 序列与参考实现完全相同。
COPIED_SOURCE = '''
"""Target module that illegally embeds reference tokens."""

import math


def copied_helper(semi_major_axis, central_mass):
    numerator = 2.0 * math.pi * (semi_major_axis ** 1.5)
    denominator = math.sqrt(6.674e-11 * central_mass)
    return numerator / denominator
'''

#: 参考实现中的"Prompt"字符串（测试自造，用于模拟受保护的表达）。
REFERENCE_PROMPT = (
    "Summarize the supplied article excerpts and cite every factual claim "
    "with the identifier of the chunk that supports it."
)

#: 目标实现中高度相似的字符串（改了几个词，但仍然 >= 0.6）。
TARGET_PROMPT = (
    "Summarize the supplied article excerpts and cite every factual claim "
    "with the identifier of the passage that supports it."
)


def write_file(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def init_reference_repo(root: Path, files: dict[str, str], message: str = "fixture") -> str:
    """把 ``files`` 提交进一个临时 git 仓库，返回 HEAD commit hash。"""
    env_args = [
        "git",
        "-c",
        "user.email=audit@example.invalid",
        "-c",
        "user.name=Audit Fixture",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative, content in files.items():
        write_file(root, relative, content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run([*env_args, "commit", "-q", "-m", message], cwd=root, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def run_audit(target: Path, reference: Path, report: Path, *extra: str) -> int:
    return audit.main(
        [
            "--target",
            str(target),
            "--reference",
            str(reference),
            "--report",
            str(report),
            *extra,
        ]
    )


def check_row(report_text: str, prefix: str) -> str:
    """取出结论摘要表中某一检查项所在的行。"""
    for line in report_text.splitlines():
        if line.startswith("| ") and prefix in line:
            return line
    raise AssertionError(f"结论摘要表中找不到检查项：{prefix}")


class AuditTestCase(unittest.TestCase):
    """公共脚手架：一次性建立目标树 / 参考树 / 报告路径。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="similarity-audit-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.target = self.root / "target"
        self.reference = self.root / "reference"
        self.report = self.root / "out" / "AUDIT.md"
        self.target.mkdir()
        self.reference.mkdir()

    def audit(self, *extra: str) -> tuple[int, str]:
        code = run_audit(self.target, self.reference, self.report, *extra)
        text = self.report.read_text(encoding="utf-8") if self.report.exists() else ""
        return code, text


class TestCleanTrees(AuditTestCase):
    def test_disjoint_trees_pass_every_check(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(
            self.target,
            "pyproject.toml",
            '[project]\nname = "scitrace"\nversion = "0.1.0"\ndependencies = ["pydantic>=2"]\n',
        )

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertNotIn("❌", report)
        for prefix in ("1. Token n-gram", "2. 逐行精确复制", "3. 字符串字面量", "4. 依赖审计", "5. 资产"):
            self.assertIn("✅", check_row(report, prefix), report)

    def test_report_parent_directory_is_created(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)

        code, _ = self.audit()

        self.assertEqual(code, 0)
        self.assertTrue(self.report.is_file())
        self.assertEqual(self.report.parent.name, "out")


class TestNgramCheck(AuditTestCase):
    def test_copied_token_sequence_fails_ngram_check(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(self.target, "src/leaked.py", COPIED_SOURCE)

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "1. Token n-gram"))
        self.assertIn("leaked.py", report)

    def test_ngram_size_is_configurable(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)

        code, report = self.audit("--ngram", "24")

        self.assertEqual(code, 0, report)
        self.assertIn("n=24", report)


class TestStringCheck(AuditTestCase):
    def test_similar_string_literal_fails_string_check(self) -> None:
        init_reference_repo(
            self.reference,
            {"src/prompts.py": f'PROMPT = "{REFERENCE_PROMPT}"\n'},
        )
        write_file(
            self.target,
            "src/recipes.py",
            TARGET_SOURCE + f'\nINSTRUCTION = "{TARGET_PROMPT}"\n',
        )

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "3. 字符串字面量"))

    def test_identical_string_literal_is_reported(self) -> None:
        init_reference_repo(
            self.reference,
            {"src/prompts.py": f'PROMPT = "{REFERENCE_PROMPT}"\n'},
        )
        write_file(self.target, "src/recipes.py", f'INSTRUCTION = "{REFERENCE_PROMPT}"\n')

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("1.000", report)

    def test_factual_doi_strings_do_not_fail(self) -> None:
        """DOI / URL 这类事实性短串天然相似，不应被判为表达抄袭。"""
        init_reference_repo(
            self.reference,
            {"tests/test_doi.py": 'KNOWN = "https://doi.org/10.31224/4087"\n'},
        )
        write_file(
            self.target,
            "tests/test_source.py",
            'SAMPLE = "https://doi.org/10.1234/ABC"\n',
        )

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertIn("✅", check_row(report, "3. 字符串字面量"))

    def test_prohibited_identifier_in_code_fails(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", 'CITATION_PREFIX = "pqac-42"\n')

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "3. 字符串字面量"))

    def test_prohibited_identifier_in_comment_is_only_a_warning(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(
            self.target,
            "src/recipes.py",
            '"""Independent module.\n\nDesign inspired by PaperQA2 (arXiv:2409.13740); code is original.\n"""\n'
            + TARGET_SOURCE,
        )

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertIn("说明性提及", report)

    def test_strict_identifiers_mode_fails_on_comment_mention(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", "# paperqa compatible naming only\nX_VALUE = 1\n")

        code, report = self.audit("--strict-identifiers")

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "3. 字符串字面量"))


class TestDependencyCheck(AuditTestCase):
    def test_banned_dependency_in_pyproject_fails(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(
            self.target,
            "pyproject.toml",
            '[project]\nname = "scitrace"\nversion = "0.1.0"\n'
            'dependencies = ["paper-qa>=5.0"]\n',
        )

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "4. 依赖审计"))

    def test_banned_import_fails(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", "import fhaviary\n\n" + TARGET_SOURCE)

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "4. 依赖审计"))

    def test_lmi_is_only_a_warning(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", "import lmi\n\n" + TARGET_SOURCE)

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertIn("✅", check_row(report, "4. 依赖审计"))
        self.assertIn("lmi", report)

    def test_separator_variants_of_banned_names_are_detected(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", "import paper_qa_docling\n\n" + TARGET_SOURCE)
        write_file(
            self.target,
            "requirements.txt",
            "paper_qa_docling==1.0\nfhaviary>=0.8\n",
        )

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "4. 依赖审计"))

    def test_unrelated_ldp_prefixed_package_is_not_banned(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(self.target, "requirements.txt", "ldpc>=2.0\n")

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertIn("✅", check_row(report, "4. 依赖审计"))

    def test_requirements_file_is_scanned(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "requirements-dev.txt", "pytest\nldp>=1.0\n")

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "4. 依赖审计"))


class TestAssetCheck(AuditTestCase):
    def test_copied_asset_hash_fails(self) -> None:
        shared_asset = "column_a,column_b\nfirst,second\n"
        init_reference_repo(
            self.reference,
            {"src/orbits.py": REFERENCE_SOURCE, "tests/stub_data/manifest.csv": shared_asset},
        )
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(self.target, "tests/stub_data/manifest.csv", shared_asset)

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "5. 资产"))
        self.assertIn("manifest.csv", report)

    def test_doc_ngram_overlap_fails(self) -> None:
        paragraph = (
            "The retrieval pipeline ranks candidate passages, merges duplicate "
            "contexts, and then answers the question with inline citations for "
            "every factual claim that the language model produces.\n"
        )
        init_reference_repo(self.reference, {"README.md": paragraph})
        write_file(self.target, "docs/design.md", paragraph)

        code, report = self.audit()

        self.assertEqual(code, 1)
        self.assertIn("❌", check_row(report, "5. 资产"))


class TestGitSnapshot(AuditTestCase):
    def test_untracked_and_modified_worktree_files_are_ignored(self) -> None:
        """只有 HEAD 中 git 跟踪的内容才是比对基准。"""
        head_source = '"""Tracked upstream module."""\n\nUPSTREAM_VALUE = 4321\n'
        head = init_reference_repo(self.reference, {"src/tracked.py": head_source})

        # 用户在上游仓库里新增的未跟踪文件 + 对跟踪文件的工作区改动。
        leaked = write_file(self.target, "src/copy_of_untracked.py", "UNTRACKED_MARKER = 'abc'\n")
        write_file(self.reference, "src/untracked_notes.py", leaked.read_text(encoding="utf-8"))
        write_file(self.reference, "src/tracked.py", leaked.read_text(encoding="utf-8"))

        code, report = self.audit()

        self.assertEqual(code, 0, report)
        self.assertIn(head, report)
        self.assertNotIn("untracked_notes.py", report.split("## 1.")[1].split("## 2.")[0])

    def test_reference_without_git_is_rejected(self) -> None:
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)

        code, report = self.audit()

        self.assertEqual(code, 2)
        self.assertEqual(report, "")


class TestArgumentHandling(AuditTestCase):
    def test_missing_target_returns_exit_code_two(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})

        code = audit.main(
            [
                "--target",
                str(self.root / "does-not-exist"),
                "--reference",
                str(self.reference),
                "--report",
                str(self.report),
            ]
        )

        self.assertEqual(code, 2)

    def test_invalid_threshold_returns_exit_code_two(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)

        code, _ = self.audit("--fail-threshold", "3.5")

        self.assertEqual(code, 2)

    def test_custom_fail_threshold_is_honoured(self) -> None:
        init_reference_repo(self.reference, {"src/orbits.py": REFERENCE_SOURCE})
        write_file(self.target, "src/recipes.py", TARGET_SOURCE)
        write_file(self.target, "src/leaked.py", COPIED_SOURCE)

        strict, _ = self.audit("--fail-threshold", "0.0001", "--file-threshold", "0.0001")
        lenient, lenient_report = self.audit(
            "--fail-threshold",
            "0.99",
            "--file-threshold",
            "0.99",
            "--max-copied-lines",
            "999",
            "--min-copied-block",
            "999",
        )

        self.assertEqual(strict, 1)
        self.assertEqual(lenient, 0, lenient_report)
        self.assertIn("99.00%", lenient_report)
        self.assertIn("✅", check_row(lenient_report, "1. Token n-gram"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
