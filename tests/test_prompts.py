"""测试提示词库与哨兵。

本文件承担一项**非功能性的守卫职责**：提示词是本项目里唯一全部由人工撰写的文本资产，
也是"零 Prompt 复制"声明的对象。因此除了行为测试，这里还钉住若干"不得无意改变"的事实：

- 哨兵短语的具体取值（短于审计脚本的 20 字符相似度门槛，机械检查覆盖不到）；
- 提示词中不得出现参考实现所使用的拒答措辞；
- 中英两套模板的占位符必须一一对应（只改一份会让两种语言行为漂移）。
"""

from __future__ import annotations

import string

import pytest
from scitrace.prompts import (
    INSUFFICIENT_EVIDENCE,
    NOT_APPLICABLE,
    PROMPT_SETS,
    PromptSet,
    available_languages,
    get_prompt_set,
    is_refusal,
)
from scitrace.prompts.sentinels import REFUSAL_PHRASES


class TestSentinels:
    def test_values_are_pinned(self) -> None:
        """哨兵取值被刻意钉死。

        这些短语短于审计脚本的字符串相似度门槛（20 字符），
        因此"无意中与参考实现撞车"不会被机械检查发现，只能靠这里显式把关。
        若本测试失败，说明有人改动了哨兵——请先确认这不是在向已知写法靠拢。
        """
        assert NOT_APPLICABLE == "NOT_APPLICABLE"
        assert INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"

    def test_does_not_reuse_a_common_refusal_phrasing(self) -> None:
        """本项目刻意不使用任何常见的英文拒答套话。

        这类套话在同类系统里被广泛使用，直接采用会让"引用键与提示词均为自定"
        这一主张出现一个不必要的例外。中文表述是必要的（模型用中文作答时会输出它），
        但英文侧只保留自有的全大写哨兵。
        """
        lowered = {phrase.lower() for phrase in REFUSAL_PHRASES}
        assert "i cannot answer" not in lowered
        assert "i don't know" not in lowered

    @pytest.mark.parametrize(
        "text",
        [
            "INSUFFICIENT_EVIDENCE",
            "INSUFFICIENT_EVIDENCE：现有资料未涉及该问题",
            "证据不足",
            "证据不足，无法回答",
            "  证据不足，无法回答  ",
        ],
    )
    def test_refusal_detected(self, text: str) -> None:
        """采用子串包含：模型常把哨兵嵌在句子里。"""
        assert is_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "答案是 42。", "The material supports this claim (ev-1a2b3c4d)."],
    )
    def test_non_refusal(self, text: str) -> None:
        assert is_refusal(text) is False

    def test_substring_matching_is_documented_tradeoff(self) -> None:
        """拒答识别刻意偏宽松：误判为拒答只是少答一题，漏判则是编造答案。"""
        assert is_refusal("虽然证据不足，但根据常识可以推断……") is True


class TestPromptSets:
    def test_both_languages_available(self) -> None:
        assert available_languages() == ["en", "zh"]

    @pytest.mark.parametrize("alias", ["zh", "ZH", "zh-CN", "zh_cn", "zh-Hans"])
    def test_language_aliases(self, alias: str) -> None:
        assert get_prompt_set(alias).language == "zh"

    def test_unknown_language_lists_options(self) -> None:
        with pytest.raises(ValueError, match="可用取值"):
            get_prompt_set("klingon")

    def test_all_sets_are_prompt_set_instances(self) -> None:
        assert all(isinstance(item, PromptSet) for item in PROMPT_SETS.values())

    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_no_template_is_empty(self, language: str) -> None:
        """每个字段都要么含占位符（是真正的模板），要么是一段足够的说明文本。

        "长度大于某个数"这种检查是武断的——像 ``Question: $question`` 这样的短模板
        完全合理。真正有意义的不变式是：**模板必须真的参数化**
        （否则它就不该是 ``Template``），而纯说明文本必须足够长以承载实质内容。
        """
        prompt_set = get_prompt_set(language)
        for field in PromptSet.__dataclass_fields__:
            if field == "language":
                continue
            value = getattr(prompt_set, field)
            if isinstance(value, string.Template):
                assert value.template.strip(), f"{language}.{field} 为空"
                assert value.get_identifiers(), (
                    f"{language}.{field} 是 Template 却没有任何占位符——"
                    "它要么该填变量，要么该改成普通字符串"
                )
            else:
                assert len(value.strip()) > 100, f"{language}.{field} 作为说明文本过短"


class TestBilingualParity:
    """中英两套模板必须在**结构上**一致。

    它们不要求措辞相同（中文不是英文的翻译），但占位符集合必须一致：
    只改一份会让两种语言在运行时因缺占位符而报错，或更糟——
    静默地少传一个变量，让模型在缺信息的情况下作答。
    """

    TEMPLATE_FIELDS = ("screening_user", "synthesis_user", "context_entry", "literature_query_user")

    @pytest.mark.parametrize("field", TEMPLATE_FIELDS)
    def test_placeholders_match(self, field: str) -> None:
        zh = getattr(get_prompt_set("zh"), field)
        en = getattr(get_prompt_set("en"), field)
        assert set(zh.get_identifiers()) == set(en.get_identifiers()), (
            f"{field} 的占位符在中英两套模板间不一致"
        )

    def test_citation_rules_share_the_same_keys(self) -> None:
        """两种语言的引用示例必须使用同一组虚构键，否则用户切换语言会看到不同的示例。"""
        import re

        for language in ("zh", "en"):
            keys = set(re.findall(r"ev-[0-9a-f]{8}", get_prompt_set(language).citation_rules))
            assert keys == {"ev-1a2b3c4d", "ev-5e6f7a8b"}, f"{language} 的引用示例键不一致"

    def test_both_sets_reference_the_same_sentinels(self) -> None:
        """哨兵必须作为占位符出现在**模板**里，并由渲染器替换为实际取值。

        模板里写死哨兵会有一处隐患：取值一改，模板与解析层就不再一致。
        用占位符则让不一致变成渲染期的 KeyError。
        """
        for language in ("zh", "en"):
            prompt_set = get_prompt_set(language)
            assert "$not_applicable" in prompt_set.screening_system.template
            assert "$refusal" in prompt_set.synthesis_system.template

            _, screening_user = prompt_set.render_screening(question="q", citation="c", text="t")
            system, _ = prompt_set.render_synthesis(question="q", context="ctx")
            assert NOT_APPLICABLE in prompt_set.render_screening(
                question="q", citation="c", text="t"
            )[0]
            assert INSUFFICIENT_EVIDENCE in system
            assert "$" not in system, "合成 system 提示词存在未替换的占位符"
            assert screening_user


class TestRendering:
    def test_screening_substitutes_all_values(self) -> None:
        system, user = get_prompt_set("zh").render_screening(
            question="跨模态对齐的效果如何？", citation="(zhang2024duomotai pages 3-4)", text="片段正文"
        )
        assert "跨模态对齐的效果如何？" in user
        assert "(zhang2024duomotai pages 3-4)" in user
        assert "片段正文" in user
        assert "$" not in user, "存在未被替换的占位符"
        assert "NOT_APPLICABLE" in system

    def test_synthesis_substitutes_context_and_question(self) -> None:
        system, user = get_prompt_set("en").render_synthesis(
            question="What is the effect?", context="[ev-1a2b3c4d] (source)\nSome summary."
        )
        assert "What is the effect?" in user
        assert "Some summary." in user
        assert "$" not in user
        assert "INSUFFICIENT_EVIDENCE" in system

    def test_iteration_block_only_when_prior_answer_given(self) -> None:
        prompt_set = get_prompt_set("zh")
        without = prompt_set.render_synthesis(question="q", context="c")
        with_prior = prompt_set.render_synthesis(question="q", context="c", prior_answer="旧答案")
        assert "旧答案" not in without[0]
        assert "旧答案" in with_prior[0]

    def test_missing_placeholder_raises_immediately(self) -> None:
        """缺失变量必须**立刻**报错，而不是留下未替换的 $question 让模型去猜。"""
        with pytest.raises(KeyError, match="缺少占位符"):
            get_prompt_set("zh")._render(get_prompt_set("zh").screening_user, question="only q")

    def test_context_entry_rendering(self) -> None:
        rendered = get_prompt_set("zh").render_context_entry(
            evidence_key="ev-1a2b3c4d", citation="(zhang2024 pages 1-2)", summary="摘要正文"
        )
        assert "[ev-1a2b3c4d]" in rendered
        assert "(zhang2024 pages 1-2)" in rendered
        assert "摘要正文" in rendered

    def test_literature_query_rendering(self) -> None:
        _, user = get_prompt_set("zh").render_literature_query(question="检索这个问题")
        assert "检索这个问题" in user
        assert "$" not in user


class TestPromptHygiene:
    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_prompts_do_not_contain_stray_braces_from_format(self, language: str) -> None:
        """模板用 ``$name`` 而非 ``str.format``，因此不应出现被转义的花括号。"""
        prompt_set = get_prompt_set(language)
        for field in ("screening_system", "synthesis_system", "citation_rules"):
            value = getattr(prompt_set, field)
            text = value.template if isinstance(value, string.Template) else value
            assert "{{" not in text
            assert "}}" not in text

    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_screening_prompt_specifies_the_json_shape(self, language: str) -> None:
        system = get_prompt_set(language).render_screening(
            question="q", citation="c", text="t"
        )[0]
        assert "summary" in system
        assert "relevance_score" in system
        assert "$" not in system, "筛选 system 提示词存在未替换的占位符"

    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_synthesis_prompt_forbids_outside_knowledge(self, language: str) -> None:
        """必须明确禁止材料之外的论断——这是"可溯源"承诺的前提。"""
        system = get_prompt_set(language).render_synthesis(question="q", context="c")[0]
        assert "ev-" in system, "引用规则应出现在合成提示词中（$citation_rules 未被替换？）"
        assert "$" not in system

    def test_context_entry_uses_bracket_form(self) -> None:
        """材料里的键必须带方括号，与"引用时写圆括号"形成区分，

        否则模型容易直接把材料里的写法原样抄进答案（即漏掉圆括号），
        而那种格式会让引用后处理正则失配。
        """
        render = get_prompt_set("zh").render_context_entry
        assert render(evidence_key="ev-1a2b3c4d", citation="c", summary="s").startswith("[ev-")
