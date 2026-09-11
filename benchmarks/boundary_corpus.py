"""边界语料生成器：用程序构造一批"专挑毛病"的输入。

## 为什么程序生成而不是放一堆样例文件

1. **可审阅**：二进制资产无法在 code review 里被检查，而评审者看不出一个
   `weird.pdf` 里到底装了什么。程序生成让"这份语料测的是什么"直接写在代码里；
2. **可复现**：任何人跑一次脚本就能得到完全相同（哈希一致）的语料；
3. **不引入第三方资产**：项目承诺"不含任何来自上游的受著作权保护的表达"，
   而最容易被无意带进来的恰恰是测试用的论文 PDF。生成的内容从根上消除这个风险。

## 语料的组织

每份语料配一条**探针说明**，写明它想触发什么行为。这份说明会同时出现在
生成报告里，让"跑了哪些边界"一目了然，而不是靠文件名猜。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CORPUS_SPECS", "CorpusSpec", "generate_corpus", "write_questions"]


@dataclass(frozen=True)
class CorpusSpec:
    """一份边界语料。"""

    filename: str
    probe: str
    """这份语料想触发什么行为。"""
    content: bytes
    expect: str
    """期望系统的表现。"""


def _pad(text: str, *, repeat: int = 40) -> str:
    """把短文本撑到有意义的长度，使它能进入分块流程。

    边界测试要区分"内容本身触发的行为"与"内容太短进不了流程"，
    因此除刻意测极短的用例外，其余都补足长度。
    """
    return "\n\n".join([text] * repeat)


def _build_specs() -> list[CorpusSpec]:
    specs: list[CorpusSpec] = []

    # ---- 空与极小 ----
    specs.append(
        CorpusSpec(
            "01_empty.txt",
            "完全空的文件",
            b"",
            "不得崩溃；应记为可摄入但产出 0 片段，或明确跳过",
        )
    )
    specs.append(
        CorpusSpec(
            "02_whitespace.txt",
            "只有空白字符",
            "   \n\n\t\t\n   \n".encode(),
            "同上：不得产出空片段（Fragment.text 非空的校验会拦住）",
        )
    )
    specs.append(
        CorpusSpec(
            "03_tiny.txt",
            "极短内容（10 字符）",
            "短文本。".encode(),
            "可摄入，产出 1 个片段；不应因低于 min_chars 而被丢弃",
        )
    )

    # ---- 分块边界 ----
    long_no_punct = "x" * 8000
    specs.append(
        CorpusSpec(
            "04_no_punctuation.txt",
            "8000 字符无任何句末标点",
            long_no_punct.encode(),
            "必须硬切；任何片段都不得超过 max_chars（3000）",
        )
    )
    specs.append(
        CorpusSpec(
            "05_no_blank_lines.txt",
            "整篇一段、无空行分隔",
            ("This is one very long paragraph without any blank line separation. " * 120).encode(),
            "应能识别为段落并切分，不得整篇变成一个片段",
        )
    )
    specs.append(
        CorpusSpec(
            "06_sentences_only.txt",
            "只有句号、没有段落的密集句子流",
            ("这是一个测试句子。中文句号密集出现。用于检验句切分与重叠逻辑。" * 80).encode(),
            "按句边界切分，片段结尾应落在标点上",
        )
    )
    specs.append(
        CorpusSpec(
            "07_headers_only.md",
            "只有标题层级、没有正文",
            b"# A\n\n## B\n\n### C\n\n#### D\n\n##### E\n\n###### F\n",
            "可摄入但不产生实质片段；section_path 不应无限增长导致崩溃",
        )
    )
    specs.append(
        CorpusSpec(
            "08_deep_sections.md",
            "多级嵌套章节",
            b"".join(
                f"{'#' * level} Level {level}\n\nBody text for level {level} with enough content.\n\n".encode()
                for level in range(1, 7)
            )
            * 8,
            "section_path 层级正确；不得因深层嵌套而丢内容",
        )
    )

    # ---- 参考文献边界 ----
    specs.append(
        CorpusSpec(
            "09_refs_in_middle.txt",
            "参考文献在中间、之后还有附录",
            (
                "Body paragraph before the references with real content.\n\n"
                "References\n\n"
                "[1] Someone. A paper title. 2020.\n"
                "[2] Other. Another paper title. 2021.\n\n"
                "Appendix A\n\n"
                "This appendix contains substantive content that must be kept.\n\n"
                "References\n\n"
                "[3] Third list entry. 2022.\n"
            ).encode(),
            "两段参考文献都要剔除；附录内容必须保留",
        )
    )
    specs.append(
        CorpusSpec(
            "10_refs_only.txt",
            "整篇都是参考文献",
            (
                "References\n\n" + "".join(f"[{i}] Author {i}. Title {i}. Journal. 20{10+i}.\n" for i in range(60))
            ).encode(),
            "全部剔除后无内容可索引；不得崩溃",
        )
    )

    # ---- 编码边界 ----
    specs.append(
        CorpusSpec(
            "11_chinese_utf8.txt",
            "纯中文 UTF-8",
            _pad("本文提出一种跨模态对齐方法，用于科研文献的证据抽取与问答。").encode("utf-8"),
            "中文 bigram 检索可用；句边界切分正确",
        )
    )
    specs.append(
        CorpusSpec(
            "12_chinese_gb18030.txt",
            "中文 GB18030 编码",
            _pad("证据可溯源问答系统通过引用键把答案关联到原文片段。").encode("gb18030"),
            "编码探测必须识别出 GB18030；不得出现乱码",
        )
    )
    specs.append(
        CorpusSpec(
            "13_chinese_big5.txt",
            "繁体中文 Big5 编码",
            _pad("證據可溯源問答系統透過引用鍵把答案關聯到原文片段。").encode("big5"),
            "编码探测必须识别出 Big5",
        )
    )
    specs.append(
        CorpusSpec(
            "14_utf8_bom.txt",
            "带 BOM 的 UTF-8",
            "\ufeff带 BOM 的文本内容，用于检验首个字符是否被正确处理。".encode("utf-8") * 20,
            "BOM 必须被剥离，否则首个 bigram 永远匹配不上",
        )
    )
    specs.append(
        CorpusSpec(
            "15_accented.txt",
            "带重音的拉丁文本（NFKC 归一化探针）",
            _pad("Café naïve résumé Zürich — Ångström units and ﬁ ligatures.").encode("utf-8"),
            "NFKC 应把 ﬁ 连字等归一化；不得因编码异常中断",
        )
    )
    specs.append(
        CorpusSpec(
            "16_emoji.txt",
            "含 Emoji 与罕见 Unicode 字符",
            _pad("结果显著提升 🎉📊 置信区间 [0.1, 0.9]，见附录 𝕬 与 ⓐ。").encode("utf-8"),
            "不得崩溃；Emoji 不应污染索引词表",
        )
    )

    # ---- 结构异常 ----
    specs.append(
        CorpusSpec(
            "17_html_masquerade.txt",
            "HTML 伪装成 .txt",
            (
                "<html><head><title>Fake</title></head><body>"
                + "<p>Paragraph about retrieval augmented generation.</p>" * 60
                + "</body></html>"
            ).encode(),
            "按纯文本处理即可；标签会进入索引但不崩溃",
        )
    )
    specs.append(
        CorpusSpec(
            "18_table_like.txt",
            "制表符对齐的表格",
            ("Method\tPrecision\tRecall\n" + "".join(f"Method{i}\t0.{i:02d}\t0.{i:02d}\n" for i in range(60))).encode(),
            "不得因超长单行而崩溃；列数据应可被检索",
        )
    )
    specs.append(
        CorpusSpec(
            "19_weird spacing.txt",
            "文件名含空格",
            _pad("Content in a file whose name contains spaces.").encode(),
            "路径处理不得出错",
        )
    )
    specs.append(
        CorpusSpec(
            "20_中文文件名.txt",
            "非 ASCII 文件名",
            _pad("文件名为中文时的内容处理。").encode("utf-8"),
            "路径处理不得出错（Windows 来源的语料很常见）",
        )
    )
    specs.append(
        CorpusSpec(
            "21_no_extension",
            "无扩展名",
            _pad("A file with no extension at all.").encode(),
            "应被跳过（无解析器支持），且不报错",
        )
    )

    # ---- 重复与冲突 ----
    shared_doi = "doi:10.5555/boundary.2024.001\n\n"
    specs.append(
        CorpusSpec(
            "22_duplicate_doi_a.txt",
            "同 DOI 的第一个文件",
            (shared_doi + _pad("Version A of the paper body text.")).encode(),
            "两文件应归并为同一篇文献并上报",
        )
    )
    specs.append(
        CorpusSpec(
            "23_duplicate_doi_b.txt",
            "同 DOI 的第二个文件（内容不同）",
            (shared_doi + _pad("Version B has different wording entirely.")).encode(),
            "同一 source_key 下两份片段不得互相覆盖",
        )
    )

    # ---- 超长 ----
    specs.append(
        CorpusSpec(
            "24_very_long.txt",
            "约 1.2 MB 的长文档",
            ("Long document sentence number filler content for boundary testing. " * 20000).encode(),
            "峰值内存不得线性膨胀；索引应在合理时间内完成",
        )
    )

    return specs


CORPUS_SPECS: tuple[CorpusSpec, ...] = tuple(_build_specs())


def generate_corpus(target: Path) -> dict[str, object]:
    """生成边界语料，返回一份带探针说明的清单。

    Returns:
        ``{"files": [...], "total_bytes": int}``，可直接序列化为 JSON 报告。
    """
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for spec in CORPUS_SPECS:
        path = root / spec.filename
        path.write_bytes(spec.content)
        entries.append(
            {
                "file": spec.filename,
                "bytes": len(spec.content),
                "probe": spec.probe,
                "expect": spec.expect,
            }
        )
    return {"root": str(root), "count": len(entries), "total_bytes": sum(e["bytes"] for e in entries), "files": entries}


#: 边界问答探针。问题本身也是边界的一部分：
#: 空问题、超长问题、纯符号问题在真实使用中都会遇到。
BOUNDARY_QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "q-in-en",
        "question": "What method does this work propose for evidence extraction?",
        "probe": "库内可回答问题（英文）",
        "expect": "应给出带引用的答案",
    },
    {
        "id": "q-in-zh",
        "question": "本文提出的跨模态对齐方法用于什么？",
        "probe": "库内可回答问题（中文）",
        "expect": "中文语料应能被中文问题检索到",
    },
    {
        "id": "q-out",
        "question": "What is the superconducting transition temperature of hydrogen sulfide at 150 GPa?",
        "probe": "语料外问题（具体科学事实，语料中必然没有）",
        "expect": "必须拒答，且不产生任何引用",
    },
    {
        "id": "q-empty",
        "question": "",
        "probe": "空问题",
        "expect": "不得崩溃；应拒答或明确报错",
    },
    {
        "id": "q-whitespace",
        "question": "     ",
        "probe": "纯空白问题",
        "expect": "同上",
    },
    {
        "id": "q-symbols",
        "question": "?!@#$%^&*()_+-=[]{}|;:',.<>/~`",
        "probe": "纯符号问题",
        "expect": "不得崩溃；应拒答",
    },
    {
        "id": "q-long",
        "question": "请详细说明 " * 300 + "这个方法？",
        "probe": "超长问题（约 1500 字）",
        "expect": "不得崩溃；token 预算应被正确计入",
    },
    {
        "id": "q-emoji",
        "question": "这个方法的效果如何？🎉📊",
        "probe": "含 Emoji 的问题",
        "expect": "不得崩溃",
    },
)


def write_questions(path: Path) -> Path:
    """把边界问答写成 JSONL。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in BOUNDARY_QUESTIONS:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    """``python -m benchmarks.boundary_corpus --out <dir>``。"""
    import argparse

    parser = argparse.ArgumentParser(description="生成边界测试语料")
    parser.add_argument("--out", type=Path, required=True, help="语料输出目录")
    parser.add_argument("--questions", type=Path, default=None, help="问题 JSONL 输出路径")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = generate_corpus(args.out)
    if args.questions:
        write_questions(args.questions)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
    for entry in report["files"]:  # type: ignore[union-attr]
        print(f"  {entry['file']:28s} {entry['bytes']:>9,d} B  {entry['probe']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
