"""HTTP 元数据来源的共享基类。

## 为什么值得抽基类

三个来源（Crossref / Semantic Scholar / OpenAlex）的**字段命名、嵌套深度、
认证方式**完全不同，但它们的**非功能行为**惊人地一致：都要设超时、都要退避重试、
都要在失败时降级成"这个来源没有贡献"，都要用同一个阈值判断标题是否真的匹配。
把这些放在三个实现里各写一遍，最直接的后果不是代码重复，而是**策略漂移**：
某天有人把 Crossref 的阈值从 0.8 调到 0.75，另外两个来源还是 0.8，
于是"哪条元数据被采纳"取决于数据源而不是配置——这种不一致极难从结果里看出来。

因此这里的分工是：基类负责"怎么问、问失败怎么办、答案算不算数"，
子类只回答四个纯映射问题（见 :class:`HttpMetadataProvider` 的类文档）。

## 降级是第一类需求

SPEC §3.5 要求"任一 provider 失败不影响主流程"。这意味着本模块的
``lookup`` 是一条**不抛异常**的路径：网络错误、非 2xx、响应不是 JSON、
字段嵌套缺失、甚至子类映射函数自身的缺陷，都只记 warning 并返回 ``None``。
把降级责任收在这里，组合层（resolver）就不必为每个来源写一遍 try/except
——那种写法必然会漏掉某一个，而漏掉的那一个会把整次摄入拖垮。
"""

from __future__ import annotations

import abc
import asyncio
import html
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from scitrace.adapters.llm.retry import RetryPolicy, with_retries
from scitrace.domain import SourcePatch
from scitrace.domain.source import normalize_doi
from scitrace.ports import MetadataMatch
from scitrace.util.hashing import normalize_text
from scitrace.util.tokenize_zh import is_cjk_ideograph, tokenize_mixed

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "DEFAULT_TIMEOUT_S",
    "MAX_PLAUSIBLE_YEAR",
    "MIN_PLAUSIBLE_YEAR",
    "TITLE_MATCH_THRESHOLD",
    "USER_AGENT",
    "HttpMetadataProvider",
    "HttpStatusError",
    "as_bool",
    "as_int",
    "as_mappings",
    "as_text",
    "as_year",
    "dig",
    "encode_doi_path",
    "strip_markup",
    "title_similarity",
]

logger = logging.getLogger(__name__)

#: 单个请求的超时（SPEC §3.5：每个 provider 独立超时，默认 15s）。
DEFAULT_TIMEOUT_S = 15.0

#: 默认重试策略：3 次尝试、指数退避 + 抖动（SPEC §3.5）。
DEFAULT_RETRY_POLICY = RetryPolicy(attempts=3)

#: 标题匹配阈值（SPEC §3.5：token Jaccard ≥ 0.8）。
#:
#: 这个值刻意偏高：元数据补全的目标是"补全"，不是"尽量补"。
#: 一条张冠李戴的标题会污染引用、嵌入与最终答案，而漏补一个字段只是让
#: 结果回到"本地可推导"的状态——**宁可不补，也不补错**。
TITLE_MATCH_THRESHOLD = 0.8

#: 用来标识自己的 User-Agent。
#:
#: Crossref 与 OpenAlex 都按"礼貌池"（polite pool）区分匿名流量与可联系流量，
#: 一个能说明身份与用途的 UA 比默认的 ``python-httpx/x.y`` 更容易在
#: 被限流时得到人工放行，也不会让来源方误判为爬虫。
USER_AGENT = "scitrace/0.1 (academic metadata enrichment; mailto via query param)"

#: 视为可信的年份区间。用于剔除 ``date-parts: [[0]]`` 一类的占位值——
#: 把 ``year=0`` 写进 ``Source`` 比留空更糟：它会静默生成 ``author0title``
#: 这样的引用键，并让"年份"这个筛选维度失去意义。
MIN_PLAUSIBLE_YEAR = 1000
MAX_PLAUSIBLE_YEAR = 2999


class HttpStatusError(Exception):
    """携带状态码的 HTTP 错误。

    ``httpx`` 自己的 :class:`httpx.HTTPStatusError` 只有 ``response``，
    而 :func:`scitrace.adapters.llm.retry.is_retryable_error` 是按异常上的
    ``status_code`` 属性判断的。若不转换，一个 429 会被判为"不可重试"，
    于是 Semantic Scholar 无 key 时最常见的限流会**一次都不重试**就放弃
    ——这正是本类存在的唯一理由。
    """

    def __init__(self, status_code: int, url: str = "") -> None:
        super().__init__(f"HTTP {status_code} for {url}" if url else f"HTTP {status_code}")
        self.status_code = status_code
        self.url = url


class HttpMetadataProvider(abc.ABC):
    """三个公开 REST 来源共用的 HTTP 骨架。

    子类只回答四个问题，其余（超时、重试、降级、相似度过滤、置信度）
    都在基类实现一次：

    - :meth:`_fetch_by_doi`：DOI → 单个来源对象；
    - :meth:`_search_by_title`：标题 → 候选来源对象列表；
    - :meth:`_title_of`：来源对象 → 标题；
    - :meth:`_patch_from_payload`：来源对象 → 稀疏补丁。

    抽象方法用 :class:`abc.ABC` + :meth:`~abc.ABC.abstractmethod` 声明，
    而不是留给子类抛 ``NotImplementedError``。理由与降级逻辑直接相关：
    本类的 ``lookup`` 会把**一切**异常降级为 ``None``，若"方法没实现"只在
    调用时炸，那个错误会被降级逻辑吞掉，表现为"这个来源永远没有贡献"
    ——一个接线缺陷伪装成一次正常的空结果。ABC 让它在构造期就失败。

    子类必须定义 :attr:`SOURCE_NAME`（小写来源名）。
    """

    #: 来源标识，写入 ``Source.metadata_sources``。
    SOURCE_NAME: ClassVar[str] = ""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_policy: RetryPolicy | None = None,
        default_params: Mapping[str, str] | None = None,
        default_headers: Mapping[str, str] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """构造 provider。

        Args:
            client: 复用的 HTTP 客户端。**为 ``None`` 时自行创建并在**
                :meth:`aclose` **中关闭；注入的客户端不关闭**——它可能被多个
                provider 共享，谁先 ``aclose`` 谁就会把别人的连接一起掐掉。
                测试通过注入 ``MockTransport`` 的客户端来拦截请求。
            timeout_s: 单请求超时秒数。
            retry_policy: 重试策略；默认 3 次尝试。
            default_params: 附加到每个请求的查询参数（如 ``mailto``）。
                值为 ``None`` 的项会被丢弃，避免发出 ``?mailto=``。
            default_headers: 附加到每个请求的请求头（如 ``x-api-key``）。
            sleep: 重试等待函数，注入以便测试（默认 :func:`asyncio.sleep`）。

        Raises:
            ValueError: 子类未定义 :attr:`SOURCE_NAME`，或 ``timeout_s`` 非正。
        """
        if not self.SOURCE_NAME:
            raise ValueError(f"{type(self).__name__} 必须定义非空的 SOURCE_NAME")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s 必须为正数，得到 {timeout_s}")

        self._timeout_s = float(timeout_s)
        self._retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        self._default_params = {
            key: value for key, value in (default_params or {}).items() if value is not None
        }
        self._default_headers = dict(default_headers or {})
        self._sleep = sleep or asyncio.sleep
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(
                timeout=self._timeout_s,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        )

    # ------------------------------------------------------------------ 只读 --

    @property
    def name(self) -> str:
        """来源标识（小写），例如 ``"crossref"``。"""
        return self.SOURCE_NAME

    @property
    def client(self) -> httpx.AsyncClient:
        """底层 HTTP 客户端（排障与测试用；生命周期仍由持有者管理）。"""
        return self._client

    @property
    def timeout_s(self) -> float:
        """单请求超时秒数。"""
        return self._timeout_s

    # -------------------------------------------------------------- 生命周期 --

    async def aclose(self) -> None:
        """关闭**自己创建**的客户端；注入的客户端不动。"""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------ 查询主流程 --

    async def lookup(self, patch: SourcePatch) -> MetadataMatch | None:
        """按已知线索查询，永不抛出。

        线索选择是**确定性**的：有 DOI 就走精确端点且不再回退到标题检索，
        只有标题才走模糊检索并按 :data:`TITLE_MATCH_THRESHOLD` 过滤。
        不做"DOI 查不到就退回标题搜"的兜底，是因为那会让"这次结果到底是
        按什么命中的"变得依赖网络偶然性——同样的输入在不同时刻得到不同的
        书目字段，是本项目最不能接受的失效模式（SPEC §1.3 可复现性）。

        最外层刻意保留一个兜底捕获：下面每一层都已各自降级，这里是防止
        **子类实现里的缺陷**（或将来新增的字段映射忘了容错）穿透到主流程。
        端口契约是"实现不应抛出"，那么最保险的做法不是在每个调用点信任它，
        而是让公开入口成为一条不可能抛出的路径。取消（``CancelledError``）
        属于 :class:`BaseException`，不会被这里吞掉，仍然正常向上传播。
        """
        try:
            return await self._lookup(patch)
        except Exception as error:  # noqa: BLE001 — 端口契约：lookup 绝不抛出
            logger.warning(
                "%s: 查询过程出现未预期错误（%s: %s），降级为无贡献",
                self.name,
                type(error).__name__,
                error,
            )
            return None

    async def _lookup(self, patch: SourcePatch) -> MetadataMatch | None:
        """查询主流程；允许抛出，由 :meth:`lookup` 统一降级。"""
        doi = normalize_doi(patch.doi)
        title = normalize_text(patch.title or "")

        if doi:
            payload = await self._fetch_by_doi(doi)
            if not isinstance(payload, Mapping):
                logger.debug("%s: DOI %s 未命中", self.name, doi)
                return None
            # DOI 端点是标识符级精确匹配，不参与相似度竞争，故记 1.0。
            return self._match_from(payload, confidence="doi", score=1.0)

        if not title:
            logger.debug("%s: 补丁既无 DOI 也无标题，跳过", self.name)
            return None

        candidates = await self._search_by_title(title)
        best_payload: Mapping[str, Any] | None = None
        best_score = 0.0
        for candidate in candidates:
            if not isinstance(candidate, Mapping):  # 子类返回畸形项时忽略而非崩
                continue
            candidate_title = self._title_of(candidate)
            if not candidate_title:
                continue
            score = title_similarity(title, candidate_title)
            if best_payload is None or score > best_score:
                best_payload, best_score = candidate, score

        if best_payload is None or best_score < TITLE_MATCH_THRESHOLD:
            logger.debug(
                "%s: %d 个候选中最高标题相似度 %.3f < %.2f，放弃补全",
                self.name,
                len(candidates),
                best_score,
                TITLE_MATCH_THRESHOLD,
            )
            return None
        return self._match_from(best_payload, confidence="title", score=best_score)

    def _match_from(
        self, payload: Mapping[str, Any], *, confidence: str, score: float
    ) -> MetadataMatch | None:
        """把来源对象转成 :class:`MetadataMatch`；映射失败或无字段则 ``None``。"""
        try:
            patch = self._patch_from_payload(payload)
            matched_title = self._title_of(payload)
        except Exception as error:  # noqa: BLE001 — 映射缺陷不得中断主流程
            logger.warning(
                "%s: 字段映射失败（%s: %s），该结果被丢弃",
                self.name,
                type(error).__name__,
                error,
            )
            return None

        if not isinstance(patch, SourcePatch):
            logger.warning("%s: 字段映射未返回 SourcePatch，结果被丢弃", self.name)
            return None
        if not patch.provided_fields():
            # 一个什么字段都没提供的"命中"对合并层毫无价值，
            # 只会让 Source.metadata_sources 多一个没贡献的来源。
            logger.debug("%s: 响应未提供任何可用字段，视为无贡献", self.name)
            return None

        return MetadataMatch(
            patch=patch,
            provider=self.name,
            confidence=confidence,
            matched_title=matched_title,
            score=score,
        )

    # ------------------------------------------------------------ 子类接口 --

    @abc.abstractmethod
    async def _fetch_by_doi(self, doi: str) -> Mapping[str, Any] | None:
        """用 DOI 精确查询，返回来源原始对象；未命中或失败返回 ``None``。"""

    @abc.abstractmethod
    async def _search_by_title(self, title: str) -> Sequence[Mapping[str, Any]]:
        """按标题检索，返回候选来源对象列表；失败或无结果返回空序列。"""

    @abc.abstractmethod
    def _title_of(self, payload: Mapping[str, Any]) -> str | None:
        """取出来源返回的标题（供相似度比较与 ``matched_title``）。"""

    @abc.abstractmethod
    def _patch_from_payload(self, payload: Mapping[str, Any]) -> SourcePatch:
        """把来源对象映射成稀疏补丁；缺失字段留 ``None``，**不要猜**。"""

    # ------------------------------------------------------------ HTTP 细节 --

    async def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | None] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any | None:
        """GET 一个 JSON 端点；任何失败都记 warning 并返回 ``None``。

        重试只覆盖"再试一次可能不同"的失败（连接错误、超时、429、5xx）。
        **JSON 解析失败不在重试范围内**：同样的响应体再取一次也不会变成合法
        JSON，重试只会把一次失败放大成三次请求。区分这两类是复用
        :func:`~scitrace.adapters.llm.retry.with_retries` 的前提。
        """
        merged_params = {
            **self._default_params,
            **{key: value for key, value in (params or {}).items() if value is not None},
        }
        merged_headers = {**self._default_headers, **dict(headers or {})}

        async def attempt() -> httpx.Response:
            response = await self._client.get(
                url, params=merged_params, headers=merged_headers, timeout=self._timeout_s
            )
            # httpx 不会自己为非 2xx 抛错。显式转成带 status_code 的异常，
            # 重试策略才能区分"429/5xx 值得重试"与"404 重试也没用"。
            if response.status_code >= 400:
                raise HttpStatusError(response.status_code, str(response.request.url))
            return response

        try:
            response = await with_retries(
                attempt,
                policy=self._retry_policy,
                description=f"{self.name} GET {url}",
                sleep=self._sleep,
            )
        except Exception as error:  # noqa: BLE001 — SPEC §3.5：失败一律降级
            logger.warning("%s: 请求失败（%s: %s）", self.name, type(error).__name__, error)
            return None

        try:
            return response.json()
        except ValueError as error:
            logger.warning("%s: 响应不是合法 JSON（%s）", self.name, error)
            return None


# --------------------------------------------------------------------- 工具 --


def encode_doi_path(doi: str) -> str:
    """把 DOI 编成可安全放进 URL 路径的形式。

    ``#`` 与 ``?`` 必须转义：httpx 会把它们当作 fragment / query 分隔符，
    于是 ``works/10.x/a#b`` 实际请求的路径变成 ``works/10.x/a``
    ——一个**静默查错论文**的缺陷，比请求失败危险得多。
    斜杠必须保留：三个来源的 DOI 端点都用裸 ``/`` 分隔注册号与后缀。
    """
    return quote(doi, safe="/")


#: 生成相似度 token 时跳过的英文停用词。
#:
#: 刻意不复用 :data:`scitrace.domain.source._TITLE_STOPWORDS`：那份表是为
#: **引用键**服务的（要求短、稳定、可读），这里是**相似度**服务的
#: （要求对"标题里多一个 the"这类差异不敏感），两者的取舍不同，
#: 合表会让任意一方的调整都影响到另一方。长度 ≤ 2 的词另由长度规则覆盖。
_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "against",
        "all",
        "also",
        "among",
        "and",
        "another",
        "any",
        "are",
        "because",
        "been",
        "before",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "done",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "here",
        "how",
        "into",
        "its",
        "just",
        "more",
        "most",
        "not",
        "only",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "toward",
        "towards",
        "under",
        "upon",
        "using",
        "very",
        "via",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "your",
    }
)

#: 长度不超过该值的拉丁 token 直接丢弃（``of`` / ``is`` / ``ai`` 这类无区分度）。
#: **仅适用于拉丁 token**：中文按字符二元组切分，二元组本身就长 2，
#: 若把长度规则套上去，中文标题会退化成空集、相似度恒为 0。
_MIN_TOKEN_LENGTH = 2


def _title_tokens(text: str) -> frozenset[str]:
    """把标题切成用于相似度比较的 token 集合。"""
    tokens: set[str] = set()
    for token in tokenize_mixed(normalize_text(text)):
        if any(is_cjk_ideograph(char) for char in token):
            # 中文 bigram 原样保留（见 _MIN_TOKEN_LENGTH 的说明）。
            tokens.add(token)
            continue
        if len(token) <= _MIN_TOKEN_LENGTH or token in _EN_STOPWORDS:
            continue
        tokens.add(token)
    return frozenset(tokens)


def title_similarity(a: str, b: str) -> float:
    """两个标题的 token 级 Jaccard 相似度。

    算法（SPEC §3.5 指定的"rapidfuzz-free 自实现"）：

    1. 分词：英文/数字连续段整体小写成一个 token，中文按字符二元组
       （复用 :func:`~scitrace.util.tokenize_zh.tokenize_mixed`）；
    2. 丢弃英文停用词与长度 ≤ 2 的拉丁 token（中文二元组不受长度规则影响）；
    3. 返回 ``|交集| / |并集|``。

    为什么用 token 集合而不是编辑距离：元数据的差异形态是**词序调换、
    大小写、标点、副标题增删**（``... Question Answering`` vs
    ``... Question Answering: A Benchmark``），集合运算对这些天然不敏感，
    而编辑距离会被长副标题严重稀释。代价是它对词序完全不敏感
    （"A 对 B 的影响"与"B 对 A 的影响"同分）——用 0.8 的高阈值与
    DOI 优先的策略把这类风险压到可接受范围。

    Args:
        a: 标题一（例如查询侧标题）。
        b: 标题二（例如来源返回的标题）。

    Returns:
        ``0.0``–``1.0`` 的相似度；任一侧没有任何有效 token 时返回 ``0.0``
        （空标题**不能**因为"两边都为空"而被判为相似）。
    """
    left = _title_tokens(a)
    right = _title_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dig(value: Any, *path: str | int) -> Any:
    """按 ``path`` 逐层深入取值；任一层缺失或类型不符即返回 ``None``。

    元数据响应普遍是深层嵌套（``authorships[0].author.display_name``），
    且任意一层都可能是 ``null``。有了它，字段映射可以一行一个字段地写
    （``as_text(dig(payload, "primary_location", "source", "display_name"))``），
    而不必写一长串会抛 ``KeyError`` / ``TypeError`` 的链式下标——
    后者正是"字段缺失就整个来源降级"的常见成因。

    Examples:
        >>> dig({"a": [{"b": 1}]}, "a", 0, "b")
        1
        >>> dig({"a": None}, "a", "b") is None
        True
    """
    current = value
    for step in path:
        if isinstance(step, int) and not isinstance(step, bool):
            if isinstance(current, str | bytes) or not isinstance(current, Sequence):
                return None
            if not -len(current) <= step < len(current):
                return None
            current = current[step]
        else:
            if not isinstance(current, Mapping):
                return None
            current = current.get(step)
        if current is None:
            return None
    return current


def as_text(value: Any) -> str | None:
    """把候选值取成非空字符串；非字符串或全空白返回 ``None``。

    刻意**不做** ``str(value)`` 强转：响应里的字段类型偶尔会变
    （作者给成对象、venue 给成列表），静默强转会把 ``{'name': ...}``
    变成 ``"{'name': ...}"`` 这样的垃圾值直接写进书目字段。
    """
    if not isinstance(value, str):
        return None
    cleaned = normalize_text(value)
    return cleaned or None


def as_int(value: Any) -> int | None:
    """把候选值取成整数；布尔、非整数浮点与其他类型返回 ``None``。

    注意 ``bool`` 必须排除：``True`` 在 Python 里是 ``1``，
    会把 ``citation_count=True`` 变成一次引用，纯属噪声。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def as_year(value: Any) -> int | None:
    """把候选值取成可信年份；区间外返回 ``None``（见 :data:`MIN_PLAUSIBLE_YEAR`）。"""
    year = as_int(value)
    if year is None or not MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR:
        return None
    return year


def as_bool(value: Any) -> bool | None:
    """只接受真正的布尔值；``"true"`` / ``1`` 一律视为缺失。

    布尔字段在合并时是有意义的已知值（``is_oa=False`` 表示"确认不开放"），
    因此宁缺毋滥：宁可留空让下一个来源回答，也不要把字符串猜成布尔。
    """
    return value if isinstance(value, bool) else None


def as_mappings(value: Any) -> list[Mapping[str, Any]]:
    """把候选序列过滤成"每一项都一定是对象"的列表。

    来源在异常情况下会把某项返回成 ``null`` 或字符串。在这里一次性过滤，
    下游的字段映射就不必为每一项再判类型。
    """
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, Mapping)]


#: 匹配 JATS / HTML 标签的正则。
#:
#: 只匹配"``<`` 后紧跟字母或 ``/``"的内容，因此数学表达式的
#: ``a < b``（``<`` 后是空格）不会被误删——剥标签的需求不该以
#: 破坏正常文本为代价。
_MARKUP_RE = re.compile(r"</?[A-Za-z][^<>]*>")


def strip_markup(value: str | None) -> str | None:
    """剥离 JATS/HTML 标记并规整空白；结果为空时返回 ``None``。

    Crossref 的 ``abstract`` 是 JATS 片段（``<jats:p>…</jats:p>``，内部常含
    ``<jats:italic>``）。这些标签若原样进入 ``Source.abstract`` 会同时污染
    三处：提示词、引用输出、嵌入向量（嵌入模型会把"标签相同的两段文本"判为
    相似，等于给所有摘要加了一个共同的伪信号）。部分记录把标签存成转义形式
    （``&lt;jats:p&gt;``），所以先反转义再剥离。
    """
    if not value:
        return None
    cleaned = normalize_text(_MARKUP_RE.sub(" ", html.unescape(value)))
    return cleaned or None
