"""架构约束的机械校验。

`pipeline` 只依赖 `domain` / `ports` / `util` / `config`，**不依赖任何适配器实现**——
这条约束在多个模块的 docstring 里被声明过，但声明本身不会阻止某次提交把它破坏掉。
而它一旦被破坏，替换实现的能力就会在不知不觉中消失：编译能过、测试能过，
直到有人真的想换一个 PDF 解析后端时才发现业务逻辑里到处是它的引用。

因此这里用 AST 静态检查把它变成一条**可执行的**规则。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "scitrace"

#: 各层允许依赖的本项目子包（顶层名字）。
ALLOWED_INTERNAL_IMPORTS: dict[str, frozenset[str]] = {
    # 领域模型：最底层，除自身与 util 外不依赖任何东西
    "domain": frozenset({"domain", "util"}),
    "ports": frozenset({"domain", "ports", "util"}),
    "util": frozenset({"util"}),
    "config": frozenset({"config", "util"}),
    "pipeline": frozenset({"config", "domain", "pipeline", "ports", "prompts", "service", "util"}),
    "agent": frozenset(
        {"agent", "config", "domain", "pipeline", "ports", "prompts", "service", "util"}
    ),
    "service": frozenset({"domain", "service", "util"}),
    "prompts": frozenset({"prompts", "util"}),
    # 适配器层是唯一允许接触外部世界（文件、HTTP、第三方库）的地方，
    # 因此它可以依赖任何内层
    "adapters": frozenset(
        {"adapters", "config", "domain", "ports", "service", "util"}
    ),
}


def iter_modules(layer: str) -> list[Path]:
    directory = PACKAGE / layer
    return sorted(directory.rglob("*.py")) if directory.is_dir() else []


def internal_imports(path: Path) -> set[str]:
    """返回一个文件中导入的 `scitrace.*` 子包名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    layers: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules = [node.module]
        for module in modules:
            parts = module.split(".")
            if parts[0] == "scitrace" and len(parts) > 1:
                layers.add(parts[1])
    return layers


@pytest.mark.parametrize("layer", sorted(ALLOWED_INTERNAL_IMPORTS))
def test_layer_only_depends_on_allowed_layers(layer: str) -> None:
    allowed = ALLOWED_INTERNAL_IMPORTS[layer]
    violations: list[str] = []
    for path in iter_modules(layer):
        # 函数内 import 同样计入：那只是把依赖藏起来，并不消除它。
        forbidden = internal_imports(path) - allowed
        if forbidden:
            violations.append(f"{path.relative_to(PACKAGE)} → {sorted(forbidden)}")
    assert not violations, (
        f"{layer} 层出现了不允许的依赖（允许：{sorted(allowed)}）：\n" + "\n".join(violations)
    )


def test_every_layer_is_covered_by_the_rule_table() -> None:
    """新增子包时必须同时声明它的依赖规则。

    漏掉一个包不会报错，只会让该包**完全不被检查**——规则表悄悄失去覆盖面。
    """
    actual = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir() and path.name != "__pycache__" and (path / "__init__.py").exists()
    }
    missing = actual - set(ALLOWED_INTERNAL_IMPORTS)
    assert not missing, f"以下子包未在 ALLOWED_INTERNAL_IMPORTS 中声明依赖规则：{sorted(missing)}"


def test_factory_is_the_only_assembly_point() -> None:
    """只有装配层与适配器层可以 import ``scitrace.adapters``。

    这条比逐层规则更强：它直接锁住"依赖倒置"这件事本身。
    """
    allowed_prefixes = ("adapters", "factory")
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        # 顶层模块（如 factory.py）取 stem；子包取目录名
        top = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        if top in allowed_prefixes:
            continue
        if "adapters" in internal_imports(path):
            violations.append(str(relative))
    assert not violations, (
        "以下模块直接依赖了 adapters，破坏了依赖倒置（应由 factory 注入）：\n"
        + "\n".join(violations)
    )
