"""测试替身：内存版的端口实现。

存在的意义不只是"跑得快"。摄入编排的正确性主要体现在**增量语义**上
（哪些文件被跳过、哪些被重建、删除的文件有没有从索引里清掉），
而验证这些语义需要能精确控制"文件何时变化、索引里此刻有什么"。
真实向量库与全文索引都不提供这种可观测性，内存替身可以。

这些替身必须**严格满足端口契约**（含哪些参数是关键字参数、返回值的排序与
``rank`` 语义），否则测试会验证一个现实中不存在的接口。
详见 ``tests/test_package_integrity.py`` 对协议一致性的守卫。
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

from scitrace.domain import Evidence, Fragment, Source, SourceKey, SourcePatch

__all__ = [
    "FakeEmbedder",
    "FakeFullTextIndex",
    "FakeResolver",
    "FakeVectorIndex",
]


def _deterministic_vector(text: str, dimension: int = 8) -> list[float]:
    """由文本派生一个确定性的归一化向量。

    用字符编码的简单散列而非 ``hash()``：后者受 ``PYTHONHASHSEED`` 影响，
    会让测试在不同进程间不可复现——而"可复现"正是本项目要验证的性质之一。
    """
    buckets = [0.0] * dimension
    for index, char in enumerate(text):
        buckets[index % dimension] += (ord(char) % 97) / 97.0
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]


class FakeEmbedder:
    """确定性内存嵌入器。"""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_deterministic_vector(text, self._dimension) for text in texts]


class FakeVectorIndex:
    """内存向量索引：朴素点积检索 + MMR。"""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Fragment, list[float]]] = {}
        self.persist_calls = 0

    @property
    def dimension(self) -> int:
        return len(next(iter(self._items.values()))[1]) if self._items else 0

    def __len__(self) -> int:
        return len(self._items)

    async def add(self, items: Sequence[tuple[Fragment, Sequence[float]]]) -> None:
        for fragment, vector in items:
            self._items[fragment.fragment_id] = (fragment, list(vector))

    async def search(
        self,
        query: Sequence[float],
        k: int,
        *,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list:
        from scitrace.ports import ScoredFragment

        if k <= 0:
            return []
        scored = [
            (sum(a * b for a, b in zip(query, vector, strict=True)), fragment)
            for fragment, vector in self._items.values()
            if allowed_keys is None or fragment.source_key in allowed_keys
        ]
        scored.sort(key=lambda item: (-item[0], item[1].fragment_id))
        return [
            ScoredFragment(fragment=fragment, score=score, origin="dense", rank=index)
            for index, (score, fragment) in enumerate(scored[:k], start=1)
        ]

    async def mmr_search(
        self,
        query: Sequence[float],
        k: int,
        *,
        fetch_k: int,
        lambda_: float = 0.5,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list:
        # 替身只保证语义正确（k、allowed_keys、rank），不追求与真实实现一致的选点结果
        return await self.search(query, k, allowed_keys=allowed_keys)

    async def remove(self, source_key: SourceKey) -> int:
        doomed = [
            key for key, (fragment, _) in self._items.items() if fragment.source_key == source_key
        ]
        for key in doomed:
            del self._items[key]
        return len(doomed)

    def clear(self) -> None:
        self._items.clear()

    def fragments(self) -> list[Fragment]:
        return [fragment for fragment, _ in self._items.values()]

    async def persist(self) -> None:
        self.persist_calls += 1


class FakeFullTextIndex:
    """内存全文索引：子串匹配 + allowed_keys 过滤。"""

    def __init__(self) -> None:
        self._fragments: dict[str, Fragment] = {}
        self._titles: dict[str, str] = {}
        self._pending: list[Fragment] = []
        self.commit_calls = 0

    def __len__(self) -> int:
        return len(self._fragments)

    async def add(
        self,
        fragments: Sequence[Fragment],
        *,
        titles: Mapping[SourceKey, str] | None = None,
    ) -> None:
        self._pending.extend(fragments)
        if titles:
            self._titles.update(titles)

    async def search(
        self,
        query: str,
        k: int,
        *,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list:
        from scitrace.ports import ScoredFragment

        if k <= 0 or not query.strip():
            return []
        needle = query.lower()
        scored = [
            (1.0 / (1 + index), fragment)
            for index, fragment in enumerate(self._fragments.values())
            if needle in fragment.text.lower()
            and (allowed_keys is None or fragment.source_key in allowed_keys)
        ]
        return [
            ScoredFragment(fragment=fragment, score=score, origin="bm25", rank=rank)
            for rank, (score, fragment) in enumerate(scored[:k], start=1)
        ]

    async def remove(self, source_key: SourceKey) -> int:
        doomed = [key for key, item in self._fragments.items() if item.source_key == source_key]
        for key in doomed:
            del self._fragments[key]
        return len(doomed)

    async def commit(self) -> None:
        self.commit_calls += 1
        for fragment in self._pending:
            self._fragments[fragment.fragment_id] = fragment
        self._pending.clear()

    def clear(self) -> None:
        self._fragments.clear()
        self._pending.clear()

    # 测试辅助（不属于端口契约）

    def indexed_ids(self) -> set[str]:
        """当前已提交的片段 id。"""
        return set(self._fragments)

    def pending_ids(self) -> set[str]:
        """尚未提交的片段 id。"""
        return {item.fragment_id for item in self._pending}


class FakeResolver:
    """固定返回一份补丁的元数据解析器。"""

    def __init__(self, patch: SourcePatch | None = None, name: str = "fake") -> None:
        self.patch = patch or SourcePatch()
        self._name = name
        self.calls: list[SourcePatch] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def enrich(self, patch: SourcePatch) -> SourcePatch:
        self.calls.append(patch)
        return patch.fill_gaps_from(self.patch)

    async def aclose(self) -> None:
        self.closed = True


class FakeScreener:
    """按片段长度给分的证据筛选器，仅供上层测试使用。"""

    def __init__(self, threshold: int = 5, score: int = 8) -> None:
        self.threshold = threshold
        self.score = score

    @property
    def name(self) -> str:
        return "fake"

    async def screen(
        self,
        question: str,
        fragments: Sequence[Fragment],
        *,
        sources: Mapping[SourceKey, Source],
    ) -> list[Evidence]:
        from scitrace.domain.evidence import make_evidence_key

        results = []
        for fragment in fragments:
            if len(fragment.text) < self.threshold:
                continue
            source = sources.get(fragment.source_key)
            stem = source.citation_stem if source else "unknown"
            results.append(
                Evidence(
                    key=make_evidence_key(fragment.source_key, fragment.fragment_id),
                    source_key=fragment.source_key,
                    fragment_id=fragment.fragment_id,
                    summary=fragment.text[:200],
                    relevance=self.score,
                    page_label=fragment.page_label,
                    section_path=list(fragment.section_path),
                    citation=f"({stem} {fragment.page_label})".replace(" )", ")"),
                )
            )
        return results

    async def aclose(self) -> None:
        return None


class FakeLLMClient:
    """按脚本返回内容的 LLM 替身。

    刻意支持"按调用顺序返回不同内容"：筛选的容错逻辑必须能在**同一批**
    里同时覆盖正常、畸形、异常三种情况，否则测不出"单条失败不影响整批"。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        default: str = '{"summary": "默认摘要", "relevance_score": 5}',
        error: Exception | None = None,
        cost_per_call: float = 0.001,
    ) -> None:
        self._responses = list(responses or [])
        self._default = default
        self._error = error
        self._cost = cost_per_call
        self.calls: list[list] = []
        self.kwargs: list[dict] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def complete(self, messages, **kwargs):
        from scitrace.ports import LLMResponse

        self.calls.append(list(messages))
        self.kwargs.append(dict(kwargs))
        if self._error is not None:
            raise self._error
        content = self._responses.pop(0) if self._responses else self._default
        return LLMResponse(
            content=content,
            model=self.model_name,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=self._cost,
        )
