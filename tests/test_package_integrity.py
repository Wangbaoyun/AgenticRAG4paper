"""包完整性与独立性守卫。

scitrace: independence-guard

本文件存在的直接原因：``ports/__init__.py`` 曾导入五个尚未创建的协议模块，
而当时的测试全都绕过了 ``scitrace.ports``，因此 153 项测试全绿却掩盖了
一个"import 就崩"的缺陷。这类问题的共性是**只在特定导入路径上暴露**，
所以用遍历式测试兜住：

1. 枚举 ``src/scitrace`` 下的**每一个**模块并导入；
2. 校验每个包 ``__all__`` 里声明的名字真实存在（声明了不存在的名字是经典静默缺陷）；
3. 静态检查源码中不出现上游依赖与标识符（SPEC §10）。

第 3 项与 ``tools/similarity_audit.py`` 有意重叠：审计脚本需要参考仓库才能运行，
而本测试无需任何外部输入，因此能在每次 ``pytest`` 时都跑一遍。

注意文件头第二行的 ``scitrace: independence-guard`` 标记：本文件必须**包含**被禁
标识符才能检测它们，因此向审计脚本声明自己属于独立性守卫工具，
从「禁用标识符」「依赖审计」两项中豁免（见 ``tools/similarity_audit.py`` 的
``is_audit_tooling``）。标记放在头部是为了让豁免在报告中一眼可查。
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE_ROOT = SRC / "scitrace"

#: SPEC §10：禁止依赖的上游同族包。
BANNED_IMPORTS = frozenset(
    {"paperqa", "paper_qa", "fhaviary", "aviary", "fhlmi", "ldp", "lmi", "paperqa2"}
)

#: SPEC §10：禁止使用的上游标识符（大小写不敏感）。
BANNED_IDENTIFIERS = ("paperqa", "pqac")


def iter_module_names() -> list[str]:
    """枚举 ``scitrace`` 包下所有可导入模块的全名。"""
    names = ["scitrace"]
    for module_info in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="scitrace."):
        names.append(module_info.name)
    return sorted(names)


def iter_source_files() -> list[Path]:
    """枚举 ``src/scitrace`` 下的全部 Python 源文件。"""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_package_root_exists() -> None:
    assert PACKAGE_ROOT.is_dir(), f"未找到包根目录：{PACKAGE_ROOT}"


def test_every_module_imports() -> None:
    """每个模块都必须能被导入。

    这是最廉价也最有效的守卫：语法错误、缺失的兄弟模块、
    ``__init__`` 中引用了不存在的符号，都会在这里立刻暴露。
    """
    failures: list[str] = []
    for name in iter_module_names():
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 — 就是要收集所有失败
            failures.append(f"{name}: {type(error).__name__}: {error}")
    assert not failures, "以下模块无法导入：\n" + "\n".join(failures)


def test_dunder_all_names_exist() -> None:
    """``__all__`` 中声明的名字必须真实存在。

    ``__all__`` 写错名字不会报错，只会在 ``from x import *`` 时静默缺失，
    属于最难发现的一类缺陷。
    """
    failures: list[str] = []
    for name in iter_module_names():
        module = importlib.import_module(name)
        for exported in getattr(module, "__all__", []):
            if not hasattr(module, exported):
                failures.append(f"{name}.__all__ 声明了不存在的 {exported!r}")
    assert not failures, "\n".join(failures)


def test_dunder_all_is_unique_and_public() -> None:
    """``__all__`` 去重，且不导出私有名。

    刻意**不**强制字典序：``__all__`` 按"常量 → 类 → 函数"分组排列比纯字典序
    更便于人工核对，是通行的 Python 惯例。这里只钉住两条真正的不变式——
    重复项会让 API 表面含糊，导出 ``_private`` 则会让"公开 API"这件事失去意义。
    """
    for name in iter_module_names():
        module = importlib.import_module(name)
        exported = list(getattr(module, "__all__", []))
        if not exported:
            continue
        assert len(exported) == len(set(exported)), f"{name}.__all__ 存在重复项"
        # ``__version__`` 这类 dunder 是约定的公开属性，只有单下划线私有名才应被拦。
        private = [
            item
            for item in exported
            if item.startswith("_") and not (item.startswith("__") and item.endswith("__"))
        ]
        assert not private, f"{name}.__all__ 导出了私有名：{private}"


def _imported_roots(tree: ast.AST) -> set[str]:
    """收集一个 AST 中所有 import 语句的顶层包名。"""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", iter_source_files(), ids=lambda p: p.name)
def test_no_banned_third_party_imports(path: Path) -> None:
    """源码不得导入上游同族包（SPEC §10）。

    导入它们会让"自研 Agent 运行时"与"无上游依赖"两项原创声明同时失效。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending = _imported_roots(tree) & BANNED_IMPORTS
    assert not offending, f"{path} 导入了被禁用的上游包：{sorted(offending)}"


@pytest.mark.parametrize("path", iter_source_files(), ids=lambda p: p.name)
def test_no_banned_identifiers(path: Path) -> None:
    """源码不得出现上游专有标识符（SPEC §10）。

    检查的是**标识符与字符串内容**而非整份文本，因为注释里说明
    "本项与 PaperQA2 的差异"是合法且必要的（PROVENANCE 要求如实署名）。
    真正要拦的是把它们用作变量、类名或对外可见的取值。
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in BANNED_IDENTIFIERS:
            offenders.append(f"标识符 {node.id!r} (行 {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.lower() in BANNED_IDENTIFIERS:
            offenders.append(f"属性 {node.attr!r} (行 {node.lineno})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            # 允许在文档字符串中提及上游名称（署名与差异说明需要），
            # 但禁止把它们用作实际的取值（引用键前缀、模块名、命令名）。
            if any(token in lowered for token in ("pqac-", "paper_qa", "paperqa.")):
                offenders.append(f"字符串取值 (行 {node.lineno})")

    assert not offenders, f"{path} 出现上游专有标识符：\n" + "\n".join(offenders)
