"""固定哨兵短语：模型输出与解析逻辑之间的契约。

## 为什么需要哨兵，而不是靠"判断模型有没有给出答案"

我们要求模型在**特定情形下输出一个固定字符串**，而不是让解析层去猜它的意图。
这样做的理由很具体：解析层需要的是**确定的分支**。

- 证据不相关 → 该片段必须被丢弃，评分记为 0；
- 证据不足以回答 → 必须走拒答路径，且**不得**产生任何引用。

如果靠语义判断（"模型这段话是不是在说不确定"），判断逻辑会随措辞漂移，
而且无法写测试。固定哨兵把这件事变成一次字符串比较。

## 为什么这里必须格外小心

哨兵短语**足够短，短到审计脚本的字符串相似度检查（≥20 字符）覆盖不到**。
也就是说，"无意中用了与参考实现相同的哨兵"这件事不会被机械检查发现，
只能靠人在这里显式地做出区分。因此本模块刻意选用与任何常见写法都不同的措辞，
并在测试中钉住具体取值（见 ``tests/test_prompts.py``）——值一旦被改动，测试会失败，
从而强制改动者重新审视这个决定。
"""

from __future__ import annotations

__all__ = [
    "INSUFFICIENT_EVIDENCE",
    "NOT_APPLICABLE",
    "REFUSAL_PHRASES",
    "is_refusal",
]

#: 片段与问题无关时，摘要字段应返回的标记。
#:
#: 选全大写 ASCII 而非自然语言短语：它不会与正常摘要的用词混淆，
#: 也让"模型照抄了提示词里的示例"这类情况一眼可见。
NOT_APPLICABLE = "NOT_APPLICABLE"

#: 证据不足以回答时使用的主哨兵（英文场景）。
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

#: 全部被认可的拒答表述。
#:
#: 中文表述单独列出，而不只是把英文哨兵翻译过去：模型在使用中文回答时，
#: 即使被明确要求输出英文哨兵，仍有一定概率输出对应的中文短语。
#: 与其和模型较劲，不如把这两种都认下来——**拒答识别宁可宽松**：
#: 误判为拒答的代价是"少回答一个问题"，漏判的代价是"编造一个答案"。
REFUSAL_PHRASES: frozenset[str] = frozenset(
    {
        INSUFFICIENT_EVIDENCE,
        "证据不足",
        "证据不足，无法回答",
        "无法回答",
    }
)


def is_refusal(text: str) -> bool:
    """判断模型输出是否表达了"证据不足，无法回答"。

    判定采用**子串包含**而非完全相等：模型常把哨兵嵌在句子里
    （例如"INSUFFICIENT_EVIDENCE：现有资料未涉及该问题"）。

    Args:
        text: 模型输出的答案正文。

    Returns:
        是否应视为拒答。
    """
    if not text:
        return False
    stripped = text.strip()
    return any(phrase in stripped for phrase in REFUSAL_PHRASES)
