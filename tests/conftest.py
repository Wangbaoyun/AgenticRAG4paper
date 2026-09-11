"""共享测试夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让测试无需安装即可导入 src 布局下的包。
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scitrace.domain import Fragment, Source  # noqa: E402


@pytest.fixture
def source() -> Source:
    """一篇元数据完整的文献。"""
    return Source(
        key="a1b2c3d4e5f60718",
        content_hash="0011223344556677",
        rel_path="papers/agentic-rag.pdf",
        title="Language Agents Achieve Superhuman Synthesis of Scientific Knowledge",
        authors=["Michael D. Skarlinski", "Sam Cox"],
        year=2024,
        doi="10.48550/arxiv.2409.13740",
        venue="arXiv",
        citation_count=42,
    )


@pytest.fixture
def source_zh() -> Source:
    """一篇中文文献，元数据不完整（无 DOI、无年份）。"""
    return Source(
        key="ffeeddccbbaa0099",
        content_hash="8899aabbccddeeff",
        rel_path="papers/中文论文.pdf",
        title="面向科研文献的证据可溯源问答方法研究",
        authors=["张三", "李四"],
    )


@pytest.fixture
def fragment() -> Fragment:
    """一个带页码的片段。"""
    return Fragment(
        fragment_id="0123456789abcdef",
        source_key="a1b2c3d4e5f60718",
        text="The system achieves 85.2% precision on the benchmark.",
        chunk_index=0,
        section_path=["4 Results"],
        page_start=5,
        page_end=6,
        char_count=54,
    )
