"""提示词：全部由本项目逐字撰写的中英双语文本资产。

本包是"零 Prompt 复制"这一主张的正面对象。关于撰写纪律与为什么必须靠人把关
而不能只依赖审计脚本，见 :mod:`scitrace.prompts.library` 与
:mod:`scitrace.prompts.sentinels` 的模块说明。
"""

from scitrace.prompts.library import PROMPT_SETS, PromptSet, available_languages, get_prompt_set
from scitrace.prompts.sentinels import (
    INSUFFICIENT_EVIDENCE,
    NOT_APPLICABLE,
    REFUSAL_PHRASES,
    is_refusal,
)

__all__ = [
    "INSUFFICIENT_EVIDENCE",
    "NOT_APPLICABLE",
    "PROMPT_SETS",
    "REFUSAL_PHRASES",
    "PromptSet",
    "available_languages",
    "get_prompt_set",
    "is_refusal",
]
