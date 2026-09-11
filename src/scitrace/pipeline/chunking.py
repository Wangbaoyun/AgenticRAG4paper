"""文本分块：把解析出的文档切成可检索、可引用的片段。

对应 SPEC §3.2 的**三级结构感知分块**。本模块是整个摄入链路里最能影响最终质量的
一步：块切得太大，检索精度下降且上下文被稀释；切得太小，证据失去上下文、引用页码
也无意义；不感知结构，则引用只能给出"第 7 页"而给不出"2.1 检索策略"。

## 为什么不是简单的定长滑窗

定长滑窗（每 N 字符切一刀）实现最简单，但在学术文献上有三个具体损失：

1. **章节信息丢失**：读者（与 LLM）无法知道一段话属于"方法"还是"结论"，
   而这两处的同一句话含义可能相反；
2. **句子被拦腰截断**：半句话的片段无论对检索还是对引用都近乎无用；
3. **参考文献区污染**：文末参考文献列表包含大量标题与作者名，与正文高度同质，
   是检索噪声的主要来源之一。

本模块依次处理这三点：先按结构切 section，再按段落聚合，超长时才在**句子边界**
切分并保留重叠。中文的句末标点与断行规则与英文不同，故句切分器同时处理两类语言。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from scitrace.config import ChunkingSettings
from scitrace.domain import Fragment, ParsedDocument
from scitrace.domain.fragment import make_fragment_id
from scitrace.util.hashing import normalize_text
from scitrace.util.tokenize_zh import is_cjk_ideograph

logger = logging.getLogger(__name__)

__all__ = [
    "TextUnit",
    "chunk_document",
    "detect_heading",
    "split_paragraphs",
    "split_sentences",
    "strip_references_section",
]

# --------------------------------------------------------------------------- #
# 句子切分
# --------------------------------------------------------------------------- #

#: 中文句末标点。中文没有词间空格，句号即句子边界，无需额外判断。
_ZH_TERMINATORS = frozenset("。！？；…")

#: 英文句末标点。句点需要额外判断（缩写、小数、人名首字母），感叹号与问号不需要。
_EN_TERMINATORS = frozenset("!?")

#: 收尾符号：句末标点后紧跟的引号/括号仍属于本句。
_CLOSERS = frozenset("\"')]}）】》」』”’")

#: 常导致误切的英文缩写（小写、不含尾点）。学术文本中高频，必须内置。
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "ch",
        "dr",
        "e.g",
        "eq",
        "etc",
        "fig",
        "i.e",
        "inc",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "pp",
        "prof",
        "ref",
        "resp",
        "sec",
        "tab",
        "vs",
        "vol",
    }
)


def _word_before(text: str, index: int) -> str:
    """取出 ``text[index]`` 之前紧邻的"词"（允许内部含点，以识别 ``e.g``）。"""
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start:index].lower().rstrip(".")


def _period_is_non_terminal(text: str, index: int) -> bool:
    """判断 ``text[index]`` 处的句点是否**不是**句子边界。

    三种情况：小数（``3.14``）、已知缩写（``et al.`` / ``e.g.``）、
    单字母人名缩写（``A. Smith``）。这三类误切在学术文本里出现频率很高，
    不处理会产生大量以缩写结尾的碎片片段。
    """
    if 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
        return True
    word = _word_before(text, index)
    if word in _ABBREVIATIONS:
        return True
    # 单字母：人名缩写或编号，如 "J. Smith" / "A."
    return len(word) == 1 and word.isalpha()


def _is_sentence_starter(char: str) -> bool:
    """句末标点之后的首个非空白字符是否像新句子的开头。

    以大写字母、数字、CJK 字符或起始引号开头 → 认为是新句子；
    以小写字母开头通常意味着上一处标点并非真正的句末（如缩写后的续写）。
    """
    return char.isupper() or char.isdigit() or is_cjk_ideograph(char) or char in "\"'“「（("


def split_sentences(text: str) -> list[str]:
    """按中英句末标点切分句子。

    同时处理中文（``。！？；…``）与英文（``.!?``）的句末标点，
    并对英文句点做缩写 / 小数 / 人名首字母的排除判断。

    Args:
        text: 待切分文本。段内换行会被视作空白。

    Returns:
        句子列表，已去除首尾空白；空输入返回空列表。

    Examples:
        >>> split_sentences("第一句。第二句！")
        ['第一句。', '第二句！']
        >>> split_sentences("See Fig. 3 for details. It works.")
        ['See Fig. 3 for details.', 'It works.']
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        buffer.append(char)

        if char in _ZH_TERMINATORS or char in _EN_TERMINATORS:
            index += 1
            # 吸收**连续的**句末标点：中文省略号是 "……"（两个 U+2026），
            # 英文也有 "!!" / "?!" 这类写法。不吸收会把一个句子拆成
            # "省略号…" + "…" 两个片段，后者是个无意义的标点片段。
            while index < length and (
                text[index] in _ZH_TERMINATORS or text[index] in _EN_TERMINATORS
            ):
                buffer.append(text[index])
                index += 1
            while index < length and text[index] in _CLOSERS:
                buffer.append(text[index])
                index += 1
            sentences.append("".join(buffer).strip())
            buffer = []
            continue

        if char == ".":
            if _period_is_non_terminal(text, index):
                index += 1
                continue
            index += 1
            while index < length and text[index] in _CLOSERS:
                buffer.append(text[index])
                index += 1
            lookahead = index
            while lookahead < length and text[lookahead].isspace():
                lookahead += 1
            if lookahead >= length or _is_sentence_starter(text[lookahead]):
                sentences.append("".join(buffer).strip())
                buffer = []
            continue

        index += 1

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]


# --------------------------------------------------------------------------- #
# 结构识别
# --------------------------------------------------------------------------- #

_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*|[IVXLC]+|[A-Z])(?P<sep>[.)]?)\s+(?P<title>\S.*)$"
)
_CJK_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:第\s*[一二三四五六七八九十百零\d]+\s*[章节部分]|[一二三四五六七八九十]+[、.．])\s*(?P<title>\S.*)$"
)
_HEADING_TERMINATORS = tuple("。.!?;,，")


def detect_heading(line: str, *, next_line_is_blank: bool = False) -> tuple[int, str] | None:
    """识别一行是否为章节标题。

    只认可四类**高置信度**信号，宁缺毋滥：

    1. Markdown ATX 标题（``## Method``）；
    2. 数字编号标题（``2``、``2.1``、``3.4.1``，级别 = 点分段数）；
    3. 罗马数字 / 单字母编号（``IV. Results``、``A. Proofs``）；
    4. 中文编号（``第三章``、``二、方法``）；
    5. 全大写短行（很多 PDF 的章节标题排版如此）。

    刻意**不**采用"短行 + 下一行空行即标题"这类宽松启发式：它会把摘要里的短句、
    公式行、表格标题统统误判成章节，使 ``section_path`` 变成噪声，
    反而比没有结构信息更糟。

    Args:
        line: 待判断的行。
        next_line_is_blank: 下一行是否为空行（仅用于全大写规则）。

    Returns:
        ``(level, title)``；非标题返回 ``None``。``level`` 从 1 开始。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None

    markdown = _MD_HEADING_RE.match(line)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    if stripped.endswith(_HEADING_TERMINATORS):
        return None

    cjk = _CJK_NUMBERED_HEADING_RE.match(stripped)
    if cjk:
        return 1, stripped

    numbered = _NUMBERED_HEADING_RE.match(stripped)
    if numbered:
        number = numbered.group("number")
        title = numbered.group("title")
        # 级别 = 点分段数（"2.1" → 2、"3" → 1）；罗马数字与单字母恒为 1 级。
        level = number.count(".") + 1 if number[:1].isdigit() else 1
        if numbered.group("sep"):
            return level, stripped
        # 无分隔符的编号标题（"3 Method" / "2.1 Retrieval"）。这类写法与
        # "以数字开头的正文句子"（"3 samples were used"）形式相同，
        # 必须额外要求标题首字母大写且整行不长，否则会把大量正文误判成章节。
        if (
            number[:1].isdigit()
            and title[:1].isupper()
            and len(stripped) <= 60
            and not stripped.endswith(_HEADING_TERMINATORS)
        ):
            return level, stripped

    # 全大写短行：至少 3 个字母，且不含小写字母（允许数字与空格）
    if (
        next_line_is_blank
        and 3 <= len(stripped) <= 60
        and any(char.isalpha() for char in stripped)
        and not any(char.islower() for char in stripped)
    ):
        return 1, stripped

    return None


def _join_wrapped_lines(lines: list[str]) -> str:
    """把软换行的多行合并为一段。

    PDF 抽取出的文本常把一句话拆到多行。合并规则按**语言**区分：

    - 前一行以 CJK 字符结尾、后一行以 CJK 字符开头 → 直接相连（中文不用空格分词）；
    - 其他情况 → 用单个空格相连（英文单词间必须有空格）。

    这条规则不处理会更隐蔽地出错：中文段落被强行插入空格后，bigram 切分会
    切出大量跨词的噪声 token，直接损害全文检索的召回质量。
    """
    if not lines:
        return ""
    parts: list[str] = [lines[0].strip()]
    for line in lines[1:]:
        current = line.strip()
        if not current:
            continue
        previous = parts[-1]
        if not previous:
            parts[-1] = current
            continue
        glue = "" if (is_cjk_ideograph(previous[-1]) and is_cjk_ideograph(current[0])) else " "
        parts[-1] = f"{previous}{glue}{current}"
    return normalize_text(" ".join(parts))


# --------------------------------------------------------------------------- #
# 参考文献剔除
# --------------------------------------------------------------------------- #

_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
        "参考文献",
        "引用文献",
        "文献列表",
    }
)


def _looks_like_reference_entry(line: str) -> bool:
    """判断一行是否像参考文献条目。

    必要性来自一个真实的误判：`detect_heading` 会把 "3. See Table 1b for ..."
    这类带编号的句子识别成标题。若用它作为"参考文献区块到此结束"的信号，
    区块会被过早截断；虽然那只是少过滤几条噪声，但这里仍然把它挡掉，
    让判据的语义更清晰。
    """
    stripped = line.lstrip()
    if stripped.startswith("["):
        return bool(re.match(r"\[\d+\]", stripped))
    if re.match(r"\d{1,3}\.\s", stripped):
        return bool(re.search(r"\b(?:19|20)\d{2}\b", line))
    return False


def _reference_blocks(rows: list[tuple[str, int]]) -> list[tuple[int, int]]:
    """找出全部参考文献区块，返回 ``[(起始行, 结束行), ...]``（结束行不含）。

    ## 为什么不是"从 References 一路切到文末"

    最初的实现就是这么做的，理由是"参考文献在文章最后"。这个假设在
    **真实的论文上不成立**：实测 PaperQA2 原文（25 页）的正文只有 9 页，
    第 9 页末尾是 References，而第 12–25 页是 "8 Methods / 8.1 PaperQA
    Implementation and Parameters / 8.2 LitQA / 8.3 WikiCrow" 等**实质性附录**。
    按原实现会静默丢弃全文 60% 的内容——而单元测试用的是合成文档
    （参考文献永远在最后），完全测不到这一点。

    现在的规则是：区块从"独立的 References 标题"开始，到**下一个真正的章节标题**
    为止；之后的内容照常收录。若找不到后续标题，则切到文末（保持原行为）。
    """
    blocks: list[tuple[int, int]] = []
    index = 0
    total = len(rows)
    while index < total:
        candidate = normalize_text(rows[index][0]).strip().lower().rstrip(":：.．")
        # 保护条件：标题之后必须至少还有一行非空内容，否则不予处理。
        # 早先这里写的是 "至少 4 行"，结果把**位于文末的短参考文献区块**整个漏掉——
        # 论文正文引用与附录引用分开列时，第二段列表往往就是这种短区块。
        has_content_after = any(rows[i][0].strip() for i in range(index + 1, total))
        if candidate not in _REFERENCE_HEADINGS or not has_content_after:
            index += 1
            continue
        end = index + 1
        while end < total:
            line = rows[end][0]
            next_is_blank = end + 1 < total and not rows[end + 1][0].strip()
            if detect_heading(line, next_line_is_blank=next_is_blank) and not _looks_like_reference_entry(line):
                break
            end += 1
        logger.debug("参考文献区块：行 %d–%d（%d 行）", index, end, end - index)
        blocks.append((index, end))
        index = end
    return blocks


def strip_references_section(text: str) -> str:
    """剔除文末的参考文献区块，返回其之前的文本。

    动机见模块 docstring：参考文献区与正文高度同质，是检索噪声的主要来源。
    """
    rows = [(line, 0) for line in text.splitlines()]
    blocks = _reference_blocks(rows)
    if not blocks:
        return text
    dropped = {index for start, end in blocks for index in range(start, end)}
    return "\n".join(line for index, (line, _) in enumerate(rows) if index not in dropped)


# --------------------------------------------------------------------------- #
# 分块主流程
# --------------------------------------------------------------------------- #


@dataclass
class TextUnit:
    """带位置信息的文本单元（段落级），是分块的中间表示。"""

    text: str
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    sentences: list[str] = field(default_factory=list)


def _iter_page_lines(document: ParsedDocument) -> list[tuple[str, int]]:
    """把文档摊平为 ``(行文本, 页码)`` 列表。"""
    rows: list[tuple[str, int]] = []
    for page in document.pages:
        for line in page.text.splitlines():
            rows.append((line, page.page_number))
    return rows


def _build_units(rows: list[tuple[str, int]]) -> list[TextUnit]:
    """按"章节 → 段落"两级结构把行序列组织成 :class:`TextUnit`。

    本函数**不负责**剔除参考文献区——那由 :func:`_reference_blocks` 在更上游
    一次性完成。职责分开是为了让"结构识别"与"噪声区裁剪"两件事各自可测，
    也避免同一判断在两处实现后逐渐不一致。
    """
    units: list[TextUnit] = []
    section_stack: list[tuple[int, str]] = []
    paragraph: list[str] = []
    paragraph_pages: list[int] = []

    def flush() -> None:
        if not paragraph:
            return
        text = _join_wrapped_lines(paragraph)
        if text:
            units.append(
                TextUnit(
                    text=text,
                    section_path=tuple(title for _, title in section_stack),
                    page_start=min(paragraph_pages),
                    page_end=max(paragraph_pages),
                )
            )
        paragraph.clear()
        paragraph_pages.clear()

    for index, (line, page_number) in enumerate(rows):
        next_is_blank = index + 1 < len(rows) and not rows[index + 1][0].strip()
        heading = detect_heading(line, next_line_is_blank=next_is_blank)

        if heading is not None:
            level, title = heading
            flush()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            continue

        if not line.strip():
            flush()
            continue

        paragraph.append(line)
        paragraph_pages.append(page_number)

    flush()
    return units


def _common_section_prefix(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[str, ...]:
    """两个章节路径的公共前缀。

    用于跨章节合并时给出**诚实的归属**：一个横跨 "1 Method" 与 "2 Results" 的块，
    声称它属于其中任何一节都是错的，而它们的公共前缀（可能是空）才是事实。
    """
    shared: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        shared.append(a)
    return tuple(shared)


def _merge_units(units: list[TextUnit], settings: ChunkingSettings) -> list[TextUnit]:
    """把段落聚合成目标长度的块，并合并过小的块。

    两轮处理：先按 ``target_chars`` 顺序聚合（不跨章节），再把低于 ``min_chars``
    的块与**同章节**的相邻块合并。第二轮必须限定同章节，否则会把"结论"的最后一句
    并进"参考文献"或"附录"，让引用位置失真。
    """
    merged: list[TextUnit] = []
    for unit in units:
        if (
            merged
            and merged[-1].section_path == unit.section_path
            and len(merged[-1].text) + len(unit.text) + 2 <= settings.target_chars
        ):
            previous = merged[-1]
            previous.text = f"{previous.text}\n\n{unit.text}"
            previous.page_start = _min_optional(previous.page_start, unit.page_start)
            previous.page_end = _max_optional(previous.page_end, unit.page_end)
        else:
            merged.append(
                TextUnit(
                    text=unit.text,
                    section_path=unit.section_path,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                )
            )

    # 小碎片回收。
    #
    # 最初的实现只与**同章节**的前一块合并，理由是"跨章节合并会让引用位置失真"。
    # 那个理由在半数情况下成立，但边界测试暴露了它的代价：一份由 48 个
    # "小节标题 + 一句话正文"组成的文档会产出 48 个 42 字符的碎片
    # （全语料 8.9% 的片段不足 100 字符，最短 4 字符）。
    # 42 字符的片段作为证据几乎无用，也会让检索结果被碎片刷屏。
    #
    # 现在改为**允许跨章节合并，但用公共前缀作为合并后的归属**——
    # 这既解决了碎片问题，又不会谎称内容属于其中某一节。
    # 另加 max_chars 上限：回收碎片不该反过来造出超长块。
    compact: list[TextUnit] = []
    for unit in merged:
        previous = compact[-1] if compact else None
        mergeable = (
            previous is not None
            and len(unit.text) < settings.min_chars
            and len(previous.text) + len(unit.text) + 2 <= settings.max_chars
        )
        if mergeable:
            assert previous is not None  # noqa: S101 - 由 mergeable 保证
            previous.text = f"{previous.text}\n\n{unit.text}"
            previous.page_end = _max_optional(previous.page_end, unit.page_end)
            previous.section_path = _common_section_prefix(
                previous.section_path, unit.section_path
            )
        else:
            compact.append(unit)
    return compact


def _min_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _max_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _split_oversized(unit: TextUnit, settings: ChunkingSettings) -> list[TextUnit]:
    """把超过 ``max_chars`` 的单元在句子边界切开，并保留尾部重叠。

    重叠的目的：一个跨越切分点的论断，其前后两半都可能与问题相关。
    没有重叠时，切分点恰好落在一个关键句上会导致两半都检索不到。
    """
    if len(unit.text) <= settings.max_chars:
        unit.sentences = split_sentences(unit.text)
        return [unit]

    sentences = split_sentences(unit.text)
    pieces: list[TextUnit] = []
    buffer: list[str] = []
    buffer_len = 0

    for sentence in sentences:
        if buffer and buffer_len + len(sentence) + 1 > settings.max_chars:
            pieces.append(
                TextUnit(
                    text=" ".join(buffer),
                    section_path=unit.section_path,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                )
            )
            # 保留尾部重叠：从缓冲末尾回取若干句，直到达到 overlap_chars
            overlap: list[str] = []
            overlap_len = 0
            for previous in reversed(buffer):
                if overlap_len >= settings.overlap_chars:
                    break
                overlap.insert(0, previous)
                overlap_len += len(previous) + 1
            buffer = overlap
            buffer_len = overlap_len

        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    if buffer:
        pieces.append(
            TextUnit(
                text=" ".join(buffer),
                section_path=unit.section_path,
                page_start=unit.page_start,
                page_end=unit.page_end,
            )
        )

    # 单句就超过 max_chars（没有句末标点的表格行、公式、乱码）：
    # 必须硬切。**max_chars 是硬上限**，不能因为"只超了一点"就放行——
    # 否则一块可能膨胀到任意大小，把它送进 LLM 上下文会直接撞上 token 上限。
    # 切分步长取 ``max_chars - overlap_chars``，使硬切也保留与其他切分一致的
    # 尾部重叠，避免关键内容恰好落在刀口上而两边都检索不到。
    result: list[TextUnit] = []
    step = max(1, settings.max_chars - settings.overlap_chars)
    for piece in pieces:
        if len(piece.text) <= settings.max_chars:
            piece.sentences = split_sentences(piece.text)
            result.append(piece)
            continue
        logger.debug("对超长单句做硬切分：长度 %d，max_chars=%d", len(piece.text), settings.max_chars)
        for offset in range(0, len(piece.text), step):
            slice_text = piece.text[offset : offset + settings.max_chars]
            if not slice_text:
                break
            result.append(
                TextUnit(
                    text=slice_text,
                    section_path=piece.section_path,
                    page_start=piece.page_start,
                    page_end=piece.page_end,
                    sentences=[slice_text],
                )
            )
    return result


def chunk_document(
    document: ParsedDocument,
    *,
    source_key: str,
    settings: ChunkingSettings | None = None,
    document_hash: str = "",
) -> list[Fragment]:
    """把解析结果切分为 :class:`Fragment` 列表。

    Args:
        document: 解析器输出（按页组织的文本）。
        source_key: 所属文献的稳定键。
        settings: 分块参数，默认取 :class:`ChunkingSettings` 的内置默认值。
        document_hash: 文档内容哈希，参与 ``fragment_id`` 派生。
            **摄入路径应当始终传入**；缺省空串只为让"不关心 id 稳定性"的
            单元测试可以省略。省略它会让"同 DOI 不同文件"的片段 id 碰撞，
            理由见 :func:`scitrace.domain.fragment.make_fragment_id`。

    Returns:
        片段列表，按 ``chunk_index`` 顺序排列。空文档返回空列表。

    Note:
        ``chunk_index`` 是**全局单调递增**的，而非章节内序号。全局序号让
        "第几个块"在整篇文档内唯一，便于按序号定位与排障。
    """
    config = settings or ChunkingSettings()
    if not document.pages:
        return []

    rows = _iter_page_lines(document)
    if config.drop_references:
        blocks = _reference_blocks(rows)
        if blocks:
            dropped = {index for start, end in blocks for index in range(start, end)}
            rows = [row for index, row in enumerate(rows) if index not in dropped]

    units = _build_units(rows)
    units = _merge_units(units, config)

    fragments: list[Fragment] = []
    chunk_index = 0
    for unit in units:
        for piece in _split_oversized(unit, config):
            text = normalize_text(piece.text)
            if not text:
                continue
            fragments.append(
                Fragment(
                    fragment_id=make_fragment_id(
                        source_key=source_key,
                        document_hash=document_hash,
                        section_path=list(piece.section_path),
                        chunk_index=chunk_index,
                    ),
                    source_key=source_key,
                    text=text,
                    chunk_index=chunk_index,
                    section_path=list(piece.section_path),
                    page_start=piece.page_start,
                    page_end=piece.page_end,
                    char_count=len(text),
                )
            )
            chunk_index += 1

    logger.info(
        "分块完成：%d 页 → %d 个片段（平均 %d 字符）",
        document.n_pages,
        len(fragments),
        (sum(item.char_count for item in fragments) // len(fragments)) if fragments else 0,
    )
    return fragments


def split_paragraphs(text: str) -> list[str]:
    """按空行切分段落，并合并段内软换行。

    供测试与上游调用方使用；``chunk_document`` 内部走的是带页码的等价流程。
    """
    blocks = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        joined = _join_wrapped_lines(lines)
        if joined:
            paragraphs.append(joined)
    return paragraphs
