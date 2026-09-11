"""测试用 PDF 生成器。

**为什么手工构造 PDF，而不是往仓库里放一个 ``paper.pdf`` 夹具：**

1. 二进制资产无法在 code review 中审阅——评审者看不出它到底装了什么；
2. 项目承诺"不含任何来自上游的受著作权保护的表达"，而最容易被无意带进来的
   恰恰是测试用的论文 PDF 与录制数据。用程序生成内容，这个风险从根上消失
   （审计脚本的资产哈希检查因此可以维持"0 命中"且不依赖人工记忆）；
3. 需要什么文本就生成什么文本，测试可读性远高于"打开某个 fixture 看第几页"。

生成的是最小合法 PDF：未压缩内容流、无交叉引用表以外的额外结构。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["build_pdf", "write_blank_pdf", "write_text_pdf"]


def _escape_pdf_text(text: str) -> bytes:
    """转义 PDF 字符串字面量中的特殊字符，并把非 Latin-1 字符替换为 ``?``。

    PDF 的 ``Tj`` 操作符使用单字节编码，中文需要嵌入 CID 字体才能正确渲染。
    本夹具只用于验证文本抽取链路，故非 Latin-1 字符统一降级为 ``?``——
    中文抽取的真实场景由端到端测试用真实 PDF 覆盖。
    """
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return escaped.encode("latin-1", errors="replace")


def build_pdf(lines: list[str], *, extra_object: bytes | None = None) -> bytes:
    """构造一个单页、含文本的 PDF。

    Args:
        lines: 每行文本，按顺序自上而下排版（行距 16pt）。
        extra_object: 可选的额外间接对象（用于构造异常 PDF 的测试）。

    Returns:
        完整的 PDF 字节串。
    """
    ops = [b"BT", b"/F1 12 Tf", b"16 TL", b"72 720 Td"]
    for index, line in enumerate(lines):
        if index:
            ops.append(b"T*")
        ops.append(b"(" + _escape_pdf_text(line) + b") Tj")
    ops.append(b"ET")
    content = b"\n".join(ops)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if extra_object is not None:
        objects.append(extra_object)

    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"

    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return header + body + xref + trailer


def write_text_pdf(path: Path, lines: list[str]) -> Path:
    """写出一个含文本的 PDF 并返回路径。"""
    Path(path).write_bytes(build_pdf(lines))
    return Path(path)


def write_blank_pdf(path: Path) -> Path:
    """写出一个**无文本层**的单页 PDF，用于模拟扫描件。"""
    content = b"BT ET"  # 空的文本块：页面存在，但没有任何可抽取文本
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"
    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    Path(path).write_bytes(header + body + xref + trailer)
    return Path(path)
