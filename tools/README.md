# tools/ — 洁净室相似度审计工具

`similarity_audit.py` 是**只依赖 Python 标准库（3.11+）**的反抄袭/相似度审计脚本，
用于支撑"`scitrace` 代码为独立编写"的声明。它把 `/home/wby/PaperQA` 的
**git HEAD 快照**当作比对基准（`git ls-files` + `git show HEAD:<path>`），因此上游工作区里
用户自己新增的未跟踪文件（`types_notes.py`、`docs/*.md`、`my_papers/`）与本地改动都不会污染基准。

## 用法

```bash
cd /home/wby/MyPaperQA
python3 tools/similarity_audit.py --target . --reference /home/wby/PaperQA \
    --report docs/AUDIT.md [--ngram 8] [--fail-threshold 0.02]
```

退出码：`0` 全部通过，`1` 存在失败项，`2` 参数/路径/git 环境错误。

## 五项检查与默认阈值

| # | 检查 | 失败判据（默认） |
| --- | --- | --- |
| 1 | token n-gram 重叠（n=8，保留注释与 docstring） | 单文件 containment > 5% 或整体 > 2% |
| 2 | 逐行精确复制（≥12 字符，忽略样板/通用惯用式） | 实质性逐字相同行 > 0，或对齐片段 ≥ 5 行 |
| 3 | 字符串字面量相似度（Prompt 泄露） | 相似度 ≥ 0.60 **且** 匹配字符 ≥ 32 即违规 |
| 4 | 依赖审计（pyproject / requirements / import） | 命中 `paperqa`、`fhaviary`、`aviary`、`fhlmi`、`ldp`；`lmi` 仅警告 |
| 5 | 资产哈希（SHA-256）+ `.md` n-gram（n=12） | 存在与上游 HEAD 同哈希的文件，或文档 containment 超阈值 |

## 自测

```bash
cd /home/wby/MyPaperQA
python3 -m unittest discover -s tests -p "test_similarity_audit.py" -v
# Ran 23 tests in 0.6s — OK
```

测试用 `tempfile` 构造迷你目录树与临时 git 仓库，覆盖：无重叠→通过、复制 n-gram→检查 1 失败、
相似字符串→检查 3 失败、`paper-qa` 依赖→检查 4 失败、专有标识符在代码 vs 注释、
搬运资产→检查 5 失败、**未跟踪文件/工作区改动被排除**、路径错误→退出码 2。

## 已知限制

脚本只能证明"没有复制表达"，不能证明"思想独立"；改写式抄袭、跨语言转译、
算法/架构层面的相似均不在检测范围内。思想层面的原创性依赖 `docs/PROVENANCE.md`
记录的洁净室流程，详见每次生成的 `docs/AUDIT.md` 中「如何解读本报告」一节。
