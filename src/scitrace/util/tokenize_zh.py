"""中文文本的免依赖切分：字符二元组（bigram）。【原创增量 ③ 的算法核心】

## 为什么不用分词器

SPEC §3.4B 要求全文索引支持中文。tantivy 不内置中文分词，而引入 jieba / pkuseg
一类的分词依赖会带来三个工程问题：

1. **词典漂移**：分词结果随词典版本变化，同一份语料在不同环境下切出不同 token，
   直接破坏"索引可复用"与"结果可复现"两项质量目标（SPEC §1.3）；
2. **未登录词**：科研文本充满新术语、缩写与专名（"跨模态对齐""稀疏专家路由"），
   而词典分词恰恰在这些词上最不可靠；
3. **体积**：中文词典通常 5–50 MB，对一个"装完即用"的检索工具是显著负担。

## bigram 的取舍

字符二元组是信息检索中的经典免词典方案::

    科研文献问答  ->  科研 研文 文献 献问 问答

- **优点**：无需词典、完全确定性、对未登录词天然鲁棒、几乎不漏召回；
- **代价**：token 数约为字符数，索引膨胀；精度略低于理想分词。

这个取舍与本项目的两段式检索架构（先召回、再用 LLM 或 Cross-Encoder 精筛，
SPEC §3.6–3.7）是配套的：**用廉价的宽召回换取鲁棒性，用重排补偿精度**。
反之，若采用精确分词，召回阶段的漏检无法被任何下游模块补救。

## 中英混排

英文/数字连续段**整体保留**为一个 token（小写化），不做词形归一——
stemming 交给索引端（tantivy 的 ``en_stem``）。本模块只负责把中英混合文本切成
"可用空格连接、可交给 whitespace tokenizer"的序列，从而让中英文共用一套索引。
"""

from __future__ import annotations

from scitrace.util.hashing import normalize_text

__all__ = ["cjk_ratio", "has_cjk", "is_cjk_ideograph", "to_index_terms", "tokenize_mixed"]

#: 视为"中文/日文/韩文表意文字"的 Unicode 区段。
#: 涵盖 CJK 扩展 A、统一表意文字、兼容表意文字、扩展 B，以及假名与谚文——
#: 这些文字都适合 bigram 处理，且都不以空格分词。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 统一表意文字
    (0xF900, 0xFAFF),  # CJK 兼容表意文字
    (0x20000, 0x2A6DF),  # CJK 扩展 B
    (0x3040, 0x30FF),  # 平假名 / 片假名
    (0xAC00, 0xD7AF),  # 谚文音节
)


def is_cjk_ideograph(char: str) -> bool:
    """判断单个字符是否属于 CJK 表意文字区段。

    Args:
        char: 长度为 1 的字符串。传入更长的字符串时只看首字符。

    Returns:
        属于任一 CJK 区段则为 ``True``。
    """
    if not char:
        return False
    code = ord(char[0])
    return any(low <= code <= high for low, high in _CJK_RANGES)


def tokenize_mixed(text: str) -> list[str]:
    """把中英混合文本切分为 token 序列。

    - CJK 连续段 → 字符 bigram（单字段落退化为该单字）；
    - 拉丁字母/数字连续段 → 整体保留并小写化；
    - 标点、空白与其他字符 → 视为边界，丢弃。

    Args:
        text: 待切分文本。本函数**不做**归一化，调用方若需要请先经
            :func:`scitrace.util.hashing.normalize_text`（:func:`to_index_terms` 已代为处理）。

    Returns:
        token 列表，按出现顺序。空输入返回空列表。

    Examples:
        >>> tokenize_mixed("Transformer 架构")
        ['transformer', '架构']
        >>> tokenize_mixed("科研文献问答")
        ['科研', '研文', '文献', '献问', '问答']
    """
    tokens: list[str] = []
    run: list[str] = []
    run_is_cjk = False

    def flush() -> None:
        if not run:
            return
        if run_is_cjk:
            if len(run) == 1:
                tokens.append(run[0])
            else:
                tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        else:
            tokens.append("".join(run).lower())
        run.clear()

    for char in text:
        if is_cjk_ideograph(char):
            if run and not run_is_cjk:
                flush()
            run_is_cjk = True
            run.append(char)
        elif char.isalnum():
            if run and run_is_cjk:
                flush()
            run_is_cjk = False
            run.append(char)
        else:
            flush()

    flush()
    return tokens


def to_index_terms(text: str) -> str:
    """把文本转换为可交给、并可由 ``whitespace`` tokenizer 还原的索引词串。

    这是索引入口/查询入口的统一转换：**同一函数**保证两侧切分完全一致，
    否则检索会因"入库与查询用了不同切分"而静默失配。

    先做 :func:`~scitrace.util.hashing.normalize_text`，使全角/半角、
    NBSP 等不可见差异不会产生不同的 token。
    """
    return " ".join(tokenize_mixed(normalize_text(text)))


def cjk_ratio(text: str) -> float:
    """统计非空白字符中 CJK 表意文字所占比例。

    用于判定语料的主导语言，进而选择更适合的分块与提示词策略
    （SPEC §9 增量 ③）。纯空白输入返回 ``0.0``。
    """
    significant = [char for char in text if not char.isspace()]
    if not significant:
        return 0.0
    cjk_count = sum(1 for char in significant if is_cjk_ideograph(char))
    return cjk_count / len(significant)


def has_cjk(text: str) -> bool:
    """文本中是否含至少一个 CJK 表意文字。"""
    return any(is_cjk_ideograph(char) for char in text)
