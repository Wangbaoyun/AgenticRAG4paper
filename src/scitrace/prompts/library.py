"""提示词库（中英双语）。

## 本模块的定位

这是本项目**唯一一处全部由人工逐字撰写**的文本资产，也是最需要说明"为何与
参考实现不同"的地方。Prompt 是典型的受著作权保护的表达：即使功能相同，
措辞、示例、约束的排列顺序都体现作者选择。因此本模块遵守两条纪律：

1. **不从任何来源改写**。全部措辞、示例引用键、约束条目顺序均为本项目自定；
2. **有意在结构上做出区分**。例如引用规则的示例键使用本项目的 ``ev-`` 方案；
   拒答哨兵见 :mod:`scitrace.prompts.sentinels`（该模块解释了为什么这里必须靠人把关，
   而不能指望审计脚本——哨兵短语短于审计的 20 字符相似度门槛）。

## 为什么用 ``string.Template`` 而不是 ``str.format``

提示词里含 JSON 示例，也就是含大量花括号。用 ``str.format`` 意味着所有字面花括号
都要写成 ``{{`` ``}}``，改一次提示词就可能多一处漏转义，而漏转义的后果是
**运行期才报错**。``string.Template`` 的 ``$name`` 占位符与花括号互不干扰。

## 双语的取舍

中文与英文各写一份，而不是"写一份英文再让模型翻译"：提示词的质量直接决定输出质量，
而机器翻译会丢失约束之间的语气差别（"必须"与"应当"在指令遵循上的效果并不相同）。
两份文本在**约束条目上严格一一对应**，由测试保证——只改一份会让两种语言的行为漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from scitrace.prompts.sentinels import INSUFFICIENT_EVIDENCE, NOT_APPLICABLE

__all__ = ["PROMPT_SETS", "PromptSet", "available_languages", "get_prompt_set"]


@dataclass(frozen=True)
class PromptSet:
    """一种语言下的全部提示词模板。

    模板用 ``$name`` 占位；调用方通过 :meth:`render` 系列方法渲染，
    缺失的占位符会**立即报错**（``Template.substitute`` 的行为），
    而不是留下一个未替换的 ``$question`` 交给模型去猜。
    """

    language: str

    # ---- 证据筛选 ----
    screening_system: Template
    screening_user: Template

    # ---- 答案合成 ----
    synthesis_system: Template
    synthesis_user: Template
    synthesis_iteration: str
    citation_rules: str
    context_entry: Template

    # ---- 检索 ----
    literature_query_system: str
    literature_query_user: Template

    def _render(self, template: Template, **values: str) -> str:
        try:
            return template.substitute(**values)
        except KeyError as error:
            raise KeyError(
                f"提示词模板缺少占位符 {error}（语言={self.language}）。"
                "模板与实际传入的变量不一致属于编码错误，应当立刻暴露。"
            ) from error

    # ---- 渲染入口 ----

    def render_screening(self, *, question: str, citation: str, text: str) -> tuple[str, str]:
        """渲染证据筛选提示词，返回 ``(system, user)``。"""
        return (
            self._render(self.screening_system, not_applicable=NOT_APPLICABLE),
            self._render(self.screening_user, question=question, citation=citation, text=text),
        )

    def render_synthesis(
        self, *, question: str, context: str, prior_answer: str | None = None
    ) -> tuple[str, str]:
        """渲染答案合成提示词，返回 ``(system, user)``。"""
        system = self._render(
            self.synthesis_system,
            citation_rules=self.citation_rules,
            refusal=INSUFFICIENT_EVIDENCE,
        )
        if prior_answer:
            system = f"{system}\n\n{self._render(self.synthesis_iteration, prior_answer=prior_answer)}"
        return system, self._render(self.synthesis_user, question=question, context=context)

    def render_context_entry(self, *, evidence_key: str, citation: str, summary: str) -> str:
        """渲染材料中的一条证据。"""
        return self._render(
            self.context_entry, evidence_key=evidence_key, citation=citation, summary=summary
        )

    def render_literature_query(self, *, question: str) -> tuple[str, str]:
        """渲染"把问题改写成检索查询"的提示词。"""
        return (
            self.literature_query_system,
            self._render(self.literature_query_user, question=question),
        )


# --------------------------------------------------------------------------- #
# 中文
# --------------------------------------------------------------------------- #

_ZH_CITATION_RULES = """\
材料中的每条证据都以 [ev-xxxxxxxx] 形式的引用键开头。引用时把键写在句末的圆括号内。

正确写法：
- 单一来源：(ev-1a2b3c4d)
- 多个来源：(ev-1a2b3c4d, ev-5e6f7a8b)

错误写法（一律不要使用）：
- ev-1a2b3c4d and ev-5e6f7a8b   —— 用连词连接，缺圆括号
- (ev-1a2b3c4d; ev-5e6f7a8b)    —— 用分号分隔
- (ev-1a2b3c4d-ev-5e6f7a8b)     —— 用连字符连成一个范围
- (参见 ev-1a2b3c4d)            —— 在键前添加说明词
- (张三等 2024)                 —— 用作者与年份代替引用键"""

_ZH = PromptSet(
    language="zh",
    screening_system=Template("""\
你是一名科研文献助理，负责判断给定的文献片段能为某个问题提供什么证据。

输出必须是一个 JSON 对象，且只包含两个字段：

- "summary"：针对该问题，从片段中提炼的**详尽**信息摘要
- "relevance_score"：1 到 10 的整数，表示该片段对回答该问题的价值

摘要的写法：
- 围绕问题筛选信息，不要面面俱到地复述整段；
- 保留具体细节——数值、单位、公式、方法名、数据集名、结论的方向与幅度；
- 直接引用原文关键短语时用引号标出，便于后续核对；
- 不要回答这个问题本身：你提供原料，不负责下结论。

评分的判断标准：
- 9–10：直接回答了问题，或提供了可直接用于回答的关键数据
- 6–8：主题相关，给出了有用的背景、方法或佐证
- 3–5：主题相邻，但对回答该问题帮助有限
- 1–2：几乎无关
- 若片段与问题完全无关：把 "summary" 设为 "$not_applicable"，"relevance_score" 设为 1

只输出 JSON 对象本身，不要输出任何其他文字，不要使用 Markdown 代码块。\
"""),
    screening_user=Template("""\
问题：$question

文献片段（来源：$citation）
---
$text
---"""),
    synthesis_system=Template("""\
你是一名严谨的科研文献综述助手。你只能依据给出的材料作答，材料之外的论断一律不允许出现。

## 引用规则
$citation_rules

## 硬性要求
1. 每一处来自材料的论断，都必须在句末标注其来源引用键；
2. 只能使用材料中**实际出现过**的引用键：不要自行编造，也不要用作者名或年份代替；
3. 若材料不足以回答问题，直接回复 $refusal，并且不要附任何引用、
   不要使用"可能""推测""尚不清楚"之类的模糊表述——明确说不知道比猜测更有价值；
4. 不要复述这些规则，直接给出答案。"""),
    synthesis_user=Template("""\
问题：$question

材料：
$context"""),
    synthesis_iteration=Template("""\
## 本轮修订要求
下面是你在上一轮给出的答案，本轮的材料集合可能已经不同。

上一轮答案：
$prior_answer

新答案只能使用**本轮材料中出现过的**引用键。上一轮引用过、但本轮材料中不存在的键，
其对应论断必须一并删除，不得保留。"""),
    citation_rules=_ZH_CITATION_RULES,
    context_entry=Template("""\
[$evidence_key] $citation
$summary"""),
    literature_query_system="""\
你负责把用户的问题改写成适合在科研文献库中检索的查询串。

改写要求：
- 提取问题中的核心概念、方法与研究对象，用它们组成查询；
- 专业术语保持原有形式：缩写、基因名、方法名、数据集名不要翻译或改写；
- 不要引入问题中未出现的概念；
- 只有当问题本身带有时间指向（如"近三年""2020 年之后"）时才填写年份范围，否则留空。

输出一个 JSON 对象，且只包含三个字段：
{"query": "检索查询串", "year_from": 年份或 null, "year_to": 年份或 null}

只输出 JSON 对象本身，不要输出任何其他文字。""",
    literature_query_user=Template("""\
问题：$question"""),
)


# --------------------------------------------------------------------------- #
# English
# --------------------------------------------------------------------------- #

_EN_CITATION_RULES = """\
Every piece of evidence in the material begins with a citation key in the form [ev-xxxxxxxx].
Cite by writing the key in parentheses at the end of the sentence.

Correct:
- single source: (ev-1a2b3c4d)
- multiple sources: (ev-1a2b3c4d, ev-5e6f7a8b)

Incorrect (never use these):
- ev-1a2b3c4d and ev-5e6f7a8b   -- joined by a conjunction, parentheses missing
- (ev-1a2b3c4d; ev-5e6f7a8b)    -- separated by a semicolon
- (ev-1a2b3c4d-ev-5e6f7a8b)     -- joined into a range with a hyphen
- (see ev-1a2b3c4d)             -- explanatory word added before the key
- (Smith et al. 2024)           -- author and year used instead of a citation key"""

_EN = PromptSet(
    language="en",
    screening_system=Template("""\
You are a research literature assistant. Your job is to determine what evidence a given
excerpt can provide for a given question.

Your output must be a single JSON object with exactly two fields:

- "summary": a **detailed** digest of the information in the excerpt that bears on the question
- "relevance_score": an integer from 1 to 10 indicating how useful the excerpt is for answering it

How to write the summary:
- Select information relative to the question; do not paraphrase the whole excerpt.
- Preserve specifics: numbers, units, equations, method names, dataset names, and the
  direction and magnitude of any reported result.
- When quoting a key phrase verbatim, put it in quotation marks so it can be checked later.
- Do not answer the question yourself. You supply raw material; you do not draw conclusions.

How to score:
- 9-10: directly answers the question, or supplies key data that can be used to answer it
- 6-8: on topic, providing useful background, method, or corroboration
- 3-5: adjacent topic, of limited use for this question
- 1-2: barely related
- If the excerpt is entirely unrelated: set "summary" to "$not_applicable" and \
  "relevance_score" to 1

Output the JSON object and nothing else. Do not wrap it in a Markdown code fence.\
"""),
    screening_user=Template("""\
Question: $question

Excerpt (source: $citation)
---
$text
---"""),
    synthesis_system=Template("""\
You are a rigorous research literature assistant. You may rely only on the material provided;
no claim outside that material is permitted.

## Citation rules
$citation_rules

## Hard requirements
1. Every claim taken from the material must carry the citation key of its source at the end
   of the sentence.
2. Use only citation keys that **actually appear** in the material. Do not invent keys, and do
   not substitute author names or years.
3. If the material is insufficient to answer, reply with exactly $refusal and attach no
   citations. Do not hedge with "possibly", "it is likely", or "remains unclear" -- stating
   plainly that you do not know is more useful than guessing.
4. Do not restate these rules. Give the answer directly."""),
    synthesis_user=Template("""\
Question: $question

Material:
$context"""),
    synthesis_iteration=Template("""\
## Revision requirements for this round
Below is the answer you produced previously. The material for this round may differ.

Previous answer:
$prior_answer

The new answer may use only citation keys that appear in **this round's** material. Any claim
whose key is absent from the current material must be removed along with the key."""),
    citation_rules=_EN_CITATION_RULES,
    context_entry=Template("""\
[$evidence_key] $citation
$summary"""),
    literature_query_system="""\
Your job is to rewrite the user's question into a query string suitable for searching a
scientific literature index.

Requirements:
- Extract the core concepts, methods, and objects of study, and build the query from them.
- Keep technical terms in their original form: do not translate or rephrase abbreviations,
  gene names, method names, or dataset names.
- Do not introduce concepts that are absent from the question.
- Fill in the year bounds only when the question itself carries a time reference
  (e.g. "in the last three years", "since 2020"); otherwise leave them null.

Output a single JSON object with exactly three fields:
{"query": "the search query", "year_from": year or null, "year_to": year or null}

Output the JSON object and nothing else.""",
    literature_query_user=Template("""\
Question: $question"""),
)


#: 语言代码 → 提示词集合。键刻意包含若干常见别名，
#: 因为语言标记在真实数据里既可能是 ``zh`` 也可能是 ``zh-CN`` / ``zh_CN``。
PROMPT_SETS: dict[str, PromptSet] = {
    "zh": _ZH,
    "zh-cn": _ZH,
    "zh_cn": _ZH,
    "zh-hans": _ZH,
    "en": _EN,
    "en-us": _EN,
    "en_us": _EN,
}


def available_languages() -> list[str]:
    """返回规范的语言代码（去重后排序）。"""
    return sorted({prompt_set.language for prompt_set in PROMPT_SETS.values()})


def get_prompt_set(language: str) -> PromptSet:
    """按语言代码取提示词集合。

    Args:
        language: 语言代码，大小写与分隔符不敏感（``zh-CN`` / ``zh_cn`` 等同）。

    Returns:
        对应的 :class:`PromptSet`。

    Raises:
        ValueError: 不支持的语言。错误信息会列出可用取值——
            配置里写错语言却只得到"key error"是没法排查的。
    """
    normalized = language.strip().lower().replace("-", "_")
    prompt_set = PROMPT_SETS.get(normalized) or PROMPT_SETS.get(normalized.replace("_", "-"))
    if prompt_set is None:
        raise ValueError(
            f"不支持的提示词语言 {language!r}；可用取值：{', '.join(sorted(PROMPT_SETS))}"
        )
    return prompt_set


# 便于测试与调用方直接引用哨兵
__all__ += ["INSUFFICIENT_EVIDENCE", "NOT_APPLICABLE"]
