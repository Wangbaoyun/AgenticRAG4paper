"""用 LLM 从论文首页推断标题。

## 为什么需要它

真实语料里大量 PDF 没有可用的书目元数据：内嵌 Title 字段为空、或等于文件名、
或写着 "Microsoft Word - 未命名文档"。这类文档的引用最终渲染成
``(anonnd01_Lewis2020 pages 2-3)`` —— **能定位到文件，但人读不出是哪篇**，
而"可溯源"的价值恰恰在于人能不能核对。

元数据来源（Crossref / OpenAlex）在这种情况下也帮不上忙：它们需要 DOI 或标题
作为查询线索，而这两者都缺。于是形成死循环：没有标题 → 查不到元数据 →
永远没有标题。

首页正文是打破这个循环唯一现成的线索，而把首屏的排版文本变成标题正是 LLM 擅长的。

## 三条刻意的约束

1. **每次摄入只调用一次**，且只发首页前 2000 字符。摄入本该是廉价的批量操作，
   给每篇文档加一次不成比例的调用会让"索引 500 篇论文"变得难以承受。
2. **默认关闭**（``metadata.llm_title_inference=false``）。它把索引从"纯本地计算"
   变成"依赖外部服务"，这个变化必须由用户显式选择。
3. **只填空缺，绝不覆盖**已有标题。Crossref 查到的标题比从首屏猜出来的可靠得多。

## 为什么它属于 pipeline 而不是 metadata 适配器

它只调用 :class:`~scitrace.ports.LLMClient` 端口，不接触任何外部世界——
没有 HTTP、没有第三方库、没有文件系统。适配器层的定义是"把外部世界翻译成领域类型"，
而这里做的是"用端口组装出一个业务步骤"，那正是 pipeline 的职责。
`tests/test_layering.py` 会拦住放错位置的模块，这条约束不是纸面声明。
"""

from __future__ import annotations

import logging

from scitrace.domain.session import Usage
from scitrace.ports import LLMClient, LLMMessage
from scitrace.prompts import PromptSet

logger = logging.getLogger(__name__)

__all__ = ["LLMTitleInferrer"]

#: 送进模型的首屏字符数上限。
#: 标题几乎总在首页最上方，多送只是浪费 token；而超长输入还会拖慢调用。
DEFAULT_MAX_CHARS = 2000

#: 标题的输出上限。标题不该超过几十个 token；给足余量但不放任模型写段落。
DEFAULT_MAX_TOKENS = 200

#: 推断出的标题超过该长度即判为可疑（模型多半在复述摘要而不是给标题）。
MAX_TITLE_CHARS = 300


class LLMTitleInferrer:
    """从首页文本推断论文标题。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptSet,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        #: 最近一次推断的用量。摄入报告会把它汇总进去——
        #: 索引阶段产生的 token 成本同样必须可见，否则"默认关闭"这个决定就无从评估。
        self.last_usage = Usage()

    async def infer(self, first_page_text: str) -> str | None:
        """从首页文本推断标题。

        Args:
            first_page_text: 首页原始文本（会被截断到 ``max_chars``）。

        Returns:
            推断出的标题；无法判断或输出不可用时返回 ``None``。

        Note:
            任何失败（调用异常、输出过长、模型明确说无法判断）都返回 ``None``
            并降级为"没有标题"。摄入不该因为一次标题推断失败而中断。
        """
        text = (first_page_text or "").strip()
        if not text:
            return None
        self.last_usage = Usage()

        system, user = self.prompts.render_title_inference(text=text[: self.max_chars])
        try:
            response = await self.llm.complete(
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except Exception as error:  # noqa: BLE001 - 推断失败不应中断摄入
            logger.warning("标题推断调用失败，将回退到文件名：%s", error)
            return None

        self.last_usage = Usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            estimated_cost=response.cost,
                            cost_currency=response.cost_currency,
                            cached_tokens=response.cached_tokens,
            cost_known=response.cost_known,
            llm_calls=1,
        )
        return _clean_title(response.content)


def _clean_title(raw: str) -> str | None:
    """清洗模型输出，取出可用标题。

    模型常见的三种不合作输出：加了引号、加了 "Title:" 前缀、以及干脆复述一整段摘要。
    前两种可以救，第三种必须拒绝——一个 500 字的"标题"塞进引用渲染只会更难看。
    """
    text = (raw or "").strip()
    if not text:
        return None
    # 去掉可能包裹的引号与常见前缀
    text = text.strip("\"'“”‘’ \n")
    for prefix in ("Title:", "标题：", "标题:", "题目：", "题目:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    # 只取第一行：模型偶尔会在标题后附一句解释
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    candidate = first_line or text
    if not candidate or len(candidate) > MAX_TITLE_CHARS:
        logger.debug("标题推断结果不可用（长度 %d），已丢弃：%r", len(candidate), candidate[:80])
        return None
    return candidate
