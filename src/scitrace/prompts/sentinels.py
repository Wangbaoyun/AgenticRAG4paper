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

#: 只有**中文**拒答表述，且它们的判定必须带位置约束（见 :func:`is_refusal`）。
#:
#: 中文表述单独列出，而不只是把英文哨兵翻译过去：模型在使用中文回答时，
#: 即使被明确要求输出英文哨兵，仍有一定概率输出对应的中文短语。
_ZH_REFUSAL_PHRASES: frozenset[str] = frozenset(
    {"证据不足", "无法回答", "没有足够的信息", "资料不足"}
)

#: 全部被认可的拒答表述。
#:
#: **只用于"提示词必须向模型声明哪些表述算拒答"这类枚举场合**。
#: 不要拿它去做子串判定——判定请用 :func:`is_refusal`，它带位置约束。
REFUSAL_PHRASES: frozenset[str] = frozenset({INSUFFICIENT_EVIDENCE}) | _ZH_REFUSAL_PHRASES

#: 判定"答案开头"的窗口长度，以字符计。
#:
#: 取值刻意压得很小，因为这里要区分的是"**声明**"与"**引用**"：
#:
#: - 真实拒答会在**第一句**就声明，实测形态是"证据不足，无法回答"（短语在 0 位）
#:   或"根据检索到的证据，证据不足……"（短语在 8 位）；
#: - 而被引用的材料可以出现在任何位置，**包括答案很靠前的地方**。
#:
#: 这个阈值最初设为 60，随即被一条回归测试打脸：把答案开头换成一句 32 字的
#: 中性陈述后，紧接着的引文落进了 60 字窗口内，一段正常答案又被判成拒答。
#: 阈值宽到能容下"开头一整句 + 引文起头"，就等于没起到位置约束的作用。
_REFUSAL_PREFIX_CHARS = 20

#: 短答案阈值：整段答案很短时，判定可以放宽到全文包含。
_SHORT_ANSWER_CHARS = 120


def is_refusal(text: str) -> bool:
    """判断模型输出是否表达了"证据不足，无法回答"。

    ## 为什么不能简单地用子串包含

    最初的实现是"包含任一拒答表述即判拒答"，理由是"误判为拒答只是少答一题，
    漏判则是编造答案"。这个推理在**只有哨兵**的时候成立，但中文表述引入了
    一个它没有覆盖的情形：**模型会引用材料里的句子，而材料里就有"证据不足"四个字**。

    实测踩到过：一次 agentic 问答产出了 2,228 字、结构完整、引用齐全的答案，
    只因正文中引用了论文原文"若证据不足，可搜索更多论文…"，
    就被整体判为拒答丢弃——`refused=True`、引用数 0、引用绑定完全没执行。
    那不是"少答一题"，那是**把最好的答案扔掉了**。

    ## 现在的判据：位置 + 长度

    - **全大写哨兵**（``INSUFFICIENT_EVIDENCE``）无歧义，出现在任何位置都算；
    - **中文表述**只在两种情形下算：出现在**答案开头**（声明的位置），
      或整段答案很短（本身就是一句声明）。
      出现在长答案中间的，按引用处理。

    Args:
        text: 模型输出的答案正文。

    Returns:
        是否应视为拒答。
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if INSUFFICIENT_EVIDENCE in stripped:
        return True
    if len(stripped) <= _SHORT_ANSWER_CHARS:
        return any(phrase in stripped for phrase in _ZH_REFUSAL_PHRASES)
    head = stripped[:_REFUSAL_PREFIX_CHARS]
    return any(phrase in head for phrase in _ZH_REFUSAL_PHRASES)
