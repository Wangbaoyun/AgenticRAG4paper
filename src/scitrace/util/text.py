"""文本抽取辅助：从正文里找出可用作元数据查询起点的标识符。

元数据补全（SPEC §3.5）需要 DOI 或标题作为查询线索。标题能从 PDF 元数据/文件名拿到，
但**DOI 几乎只能从正文里找**——绝大多数出版社的 PDF 会把 DOI 印在第一页。
在校验和清理之前先把它捞出来，能让精确查询端点的命中率大幅提高
（有 DOI 时 Crossref/OpenAlex 是 O(1) 精确查找，没有时只能走模糊检索，
准确率和限流表现都差很多）。
"""

from __future__ import annotations

import json
import re

__all__ = [
    "extract_balanced_json_object",
    "find_arxiv_id",
    "find_doi",
    "first_page_text",
    "parse_json_object",
    "strip_reasoning_tags",
]

#: DOI 的形式定义：``10.`` + 4~9 位注册机构号 + ``/`` + 后缀。
#: 后缀允许到空白/引号/尖括号为止，并在匹配后剥离尾部标点。
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)

#: DOI 后缀常被句子标点"粘"上，需要从尾部剥离。
_TRAILING_PUNCTUATION = ".,;:)]}>\"'"

#: arXiv 编号：``arXiv:2409.13740`` 或 ``arXiv:2409.13740v2``。
_ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

#: 思维链标签。推理模型会在正文里夹带思考过程，
#: 解析结构化输出前必须剥离，否则会干扰 JSON 截取。
_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)

#: Markdown 代码围栏。
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def find_doi(text: str) -> str | None:
    """从文本中找出第一个 DOI。

    Args:
        text: 待搜索文本（通常取第一页）。

    Returns:
        归一化前的 DOI 原始串（清理尾部标点后）；未找到返回 ``None``。

    Note:
        返回的是**未经归一化**的串，交由
        :func:`scitrace.domain.source.normalize_doi` 统一处理，
        避免同一件事在两处实现。
    """
    match = _DOI_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).rstrip(_TRAILING_PUNCTUATION)
    # 括号配对保护：DOI 后缀里合法地含括号（如 10.1002/(SICI)1099-...），
    # 而句子末尾的右括号不是 DOI 的一部分。两者形式相同，
    # 用"数量是否配平"来区分。
    while candidate.count(")") > candidate.count("(") and candidate.endswith(")"):
        candidate = candidate[:-1]
    return candidate or None


def find_arxiv_id(text: str) -> str | None:
    """从文本中找出第一个 arXiv 编号（不含版本号）。"""
    match = _ARXIV_RE.search(text)
    return match.group(1) if match else None


def first_page_text(pages_text: list[str], *, limit: int = 4000) -> str:
    """取第一页的开头若干字符，用于查找 DOI / arXiv 编号。

    限制长度有两个理由：DOI 只印在第一页顶部；而超长文本会显著拖慢正则匹配
    （回溯风险），何况这份文本只用于查找标识符。
    """
    if not pages_text:
        return ""
    return pages_text[0][:limit]


def strip_reasoning_tags(text: str) -> str:
    """移除推理模型的思维链标签，返回可见正文。

    Args:
        text: 模型原始输出。

    Returns:
        去除 ``<think>…</think>`` 等区块并去除首尾空白后的文本。
        未闭合的标签会被保留——那通常说明输出被截断，删掉反而丢失线索。
    """
    return _REASONING_TAG_RE.sub("", text).strip()


def extract_balanced_json_object(text: str) -> str | None:
    """从文本中截取第一个**括号配平**的 JSON 对象。

    比"取第一个 ``{`` 到最后一个 ``}``"稳健得多：后者在模型输出
    "先给一个示例 ``{...}``，然后是真正的结果 ``{...}``" 时会取到跨越两段的
    非法字符串。这里的实现跟踪字符串字面量与转义状态，逐字符定位配平点。

    Args:
        text: 待搜索文本。

    Returns:
        截取到的子串；未找到配平对象时返回 ``None``。
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


#: 常见畸形 JSON 的修复规则，按顺序应用。刻意保持**保守**：
#: 只在能明确判断意图时改写，不尝试猜测截断内容——猜测会产出
#: "看起来解析成功但内容是错的"结果，比明确失败更危险。
_JSON_REPAIR_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r",\s*([}\]])"), r"\1"),  # 尾逗号
    (re.compile(r"([{,]\s*)([A-Za-z_]\w*)\s*:"), r'\1"\2":'),  # 无引号的键
    (re.compile(r"[\u201c\u201d]"), '"'),  # 中文引号
    (re.compile(r"[\u2018\u2019]"), "'"),
)


def parse_json_object(text: str) -> dict[str, object] | None:
    """从模型输出中尽力解析出一个 JSON 对象。

    按"从严格到宽松"的顺序尝试：直接解析 → 剥离代码围栏 → 括号配平截取 →
    保守的规则化修复。全部失败返回 ``None``。

    **不抛异常**是刻意的：调用方（证据筛选、查询改写）需要的是"这块解析失败，
    跳过它"，而不是让整批处理中断。一次解析失败不该毁掉整次问答。

    Args:
        text: 模型原始输出。

    Returns:
        解析出的字典；无法解析时返回 ``None``。
    """
    if not text or not text.strip():
        return None
    cleaned = strip_reasoning_tags(text)

    fenced = _FENCE_RE.search(cleaned)
    candidates = [
        cleaned,
        fenced.group(1).strip() if fenced else "",
        extract_balanced_json_object(cleaned) or "",
    ]
    for candidate in candidates:
        parsed = _try_load_object(candidate)
        if parsed is not None:
            return parsed

    repaired = cleaned
    for pattern, replacement in _JSON_REPAIR_RULES:
        repaired = pattern.sub(replacement, repaired)
    return _try_load_object(extract_balanced_json_object(repaired) or repaired)


def _try_load_object(candidate: str) -> dict[str, object] | None:
    """尝试把候选串解析为 JSON 对象（顶层不是对象则返回 ``None``）。"""
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
