"""测试工具层的确定性哈希与中文 bigram 切分。

这两个模块是"索引可复用""结果可复现""引用可回溯"三项质量目标的地基，
因此测试重点不是常规路径，而是**边界与不变式**。
"""

from __future__ import annotations

import pytest
from scitrace.util import (
    cjk_ratio,
    derive_key,
    normalize_text,
    sha256_hex,
    stable_json,
    to_index_terms,
    tokenize_mixed,
)
from scitrace.util.hashing import hash_file


class TestDeterministicKeys:
    def test_derive_key_is_separator_safe(self) -> None:
        """拼接歧义防护：("a","bc") 与 ("ab","c") 必须产生不同的键。

        这是使用 US 分隔符而非直接字符串拼接的全部理由；若此测试失败，
        说明有人把 derive_key 改成了朴素拼接。
        """
        assert derive_key("a", "bc") != derive_key("ab", "c")

    def test_derive_key_is_stable_across_calls(self) -> None:
        assert derive_key("q", "fp") == derive_key("q", "fp")

    def test_derive_key_order_matters(self) -> None:
        assert derive_key("a", "b") != derive_key("b", "a")

    def test_derive_key_normalizes_before_hashing(self) -> None:
        """全角/半角差异不应产生不同的键，否则同一问题会得到两个 session_id。"""
        assert derive_key("ＡＢＣ") == derive_key("ABC")
        assert derive_key("a\u00a0b") == derive_key("a b")

    def test_derive_key_prefix_and_length(self) -> None:
        key = derive_key("x", prefix="ev-", length=8)
        assert key.startswith("ev-")
        assert len(key) == 3 + 8

    def test_sha256_hex_rejects_out_of_range_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            sha256_hex("x", length=0)
        with pytest.raises(ValueError, match="length"):
            sha256_hex("x", length=65)

    def test_stable_json_is_key_sorted_and_compact(self) -> None:
        assert stable_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_stable_json_preserves_non_ascii(self) -> None:
        """中文标题不应被转义成 \\uXXXX——否则同一标题有两种编码形态。"""
        assert "文献" in stable_json({"t": "文献"})

    def test_stable_json_does_not_raise_on_exotic_types(self) -> None:
        """指纹计算不应因配置里多了一个 Path/时间戳而失败。"""
        from pathlib import Path

        assert "papers" in stable_json({"p": Path("papers/x.pdf")})


class TestNormalizeText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ＡＢＣ", "ABC"),  # 全角 → 半角
            ("a\u00a0b", "a b"),  # NBSP → 空格
            ("a\u200bb", "ab"),  # 零宽空格删除
            ("a\r\nb", "a\nb"),  # CRLF → LF
            ("a\rb", "a\nb"),
            ("a   b", "a b"),  # 连续空格压缩
            ("a\n\n\n\nb", "a\n\nb"),  # 三个以上换行压缩为两个
            ("  padded  ", "padded"),
            ("", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_text(raw) == expected

    def test_does_not_fold_case_or_strip_punctuation(self) -> None:
        """大小写与标点承载语义，归一化不应越权处理。"""
        assert normalize_text("Hello, World!") == "Hello, World!"


class TestTokenizeChinese:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("科研文献", ["科研", "研文", "文献"]),
            ("中", ["中"]),  # 单字段落退化为单字
            ("Transformer 架构", ["transformer", "架构"]),
            ("BERT模型", ["bert", "模型"]),
            ("a-b", ["a", "b"]),  # 标点作边界
            ("", []),
            ("   ", []),
            ("！！！", []),  # 纯标点无 token
        ],
    )
    def test_tokenize_mixed(self, text: str, expected: list[str]) -> None:
        assert tokenize_mixed(text) == expected

    def test_bigram_recall_on_unseen_terms(self) -> None:
        """未登录术语必须仍可被 bigram 召回——这是不用词典分词的全部理由。"""
        query_terms = set(to_index_terms("跨模态对齐").split())
        doc_terms = set(to_index_terms("本文提出一种跨模态对齐方法").split())
        assert query_terms & doc_terms, "未登录术语应当有 bigram 重叠"

    def test_index_terms_roundtrip_is_whitespace_tokenizable(self) -> None:
        """to_index_terms 的输出必须能被 whitespace tokenizer 还原成同样的 token。"""
        terms = to_index_terms("证据可溯源问答 Agentic RAG")
        assert terms.split() == tokenize_mixed(normalize_text("证据可溯源问答 Agentic RAG"))

    def test_index_terms_normalizes_fullwidth(self) -> None:
        """全角/半角输入必须产生同一索引词串，否则入库与查询会静默失配。"""
        assert to_index_terms("ＡＩ模型") == to_index_terms("AI模型")

    def test_no_cross_language_token_bleed(self) -> None:
        """中英相邻不应粘成一个 token。"""
        tokens = tokenize_mixed("RAG系统")
        assert "rag" in tokens
        assert "系统" in tokens


class TestCjkRatio:
    def test_pure_chinese(self) -> None:
        assert cjk_ratio("科研文献") == pytest.approx(1.0)

    def test_pure_english(self) -> None:
        assert cjk_ratio("research paper") == pytest.approx(0.0)

    def test_mixed(self) -> None:
        ratio = cjk_ratio("RAG 检索")
        assert 0.0 < ratio < 1.0

    def test_empty_and_whitespace(self) -> None:
        assert cjk_ratio("") == pytest.approx(0.0)
        assert cjk_ratio("   \n\t") == pytest.approx(0.0)


class TestHashFile:
    def test_matches_plain_sha256_for_small_file(self, tmp_path) -> None:
        import hashlib

        target = tmp_path / "a.bin"
        payload = b"hello world" * 1000
        target.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()[:16]
        assert hash_file(target) == expected

    def test_streaming_matches_for_multichunk_file(self, tmp_path) -> None:
        """分块读取必须与整体哈希一致（chunk_size 小于文件大小）。"""
        import hashlib

        target = tmp_path / "big.bin"
        payload = bytes(range(256)) * 8192  # 2 MiB
        target.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()[:16]
        assert hash_file(target, chunk_size=4096) == expected

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "nope.bin")
