"""通用工具：纯函数、零业务依赖。

本包内的模块不得 import `scitrace` 的其他子包（domain / ports / adapters / pipeline / agent），
以保证可以被任何层安全复用。
"""

from scitrace.util.hashing import (
    derive_key,
    hash_file,
    normalize_text,
    sha256_hex,
    stable_json,
)
from scitrace.util.tokenize_zh import (
    cjk_ratio,
    is_cjk_ideograph,
    to_index_terms,
    tokenize_mixed,
)

__all__ = [
    "cjk_ratio",
    "derive_key",
    "hash_file",
    "is_cjk_ideograph",
    "normalize_text",
    "sha256_hex",
    "stable_json",
    "to_index_terms",
    "tokenize_mixed",
]
