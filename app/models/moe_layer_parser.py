"""
MoE layer frequency resolver — parse Megatron-LM / MindSpeed-LLM style
``--moe-layer-freq`` and ``--first-k-dense-replace`` arguments into
``num_moe_layers`` and ``num_dense_layers``.

Supports four input formats:

    ======================== ============================================= ==============================
    Format                    Example                                       Source
    ======================== ============================================= ==============================
    Integer N (all-MoE)      ``1``                                          Both
    Integer N (periodic)     ``3``  (1 MoE per 3 layers)                   Megatron-LM
    MindSpeed sentinel       ``-1`` (equivalent to ``1``)                  MindSpeed-LLM
    Python list expression   ``"[0]*3+[1]*58"``                            Megatron-LM
    ======================== ============================================= ==============================

``--first-k-dense-replace`` (MindSpeed-LLM extension) is handled as a
pre-processing step: the first *K* layers are forced to 0 (Dense) before
the main ``moe_layer_freq`` pattern is applied to the remaining layers.
"""

from __future__ import annotations

import re
from typing import Union

# ── Pattern validation (same approach as Megatron-LM's _eval_pattern) ──
# Only digits, commas, brackets, parentheses, + and * are allowed.
_PATTERN_SAFE_RE = re.compile(r"[^,\d\[\]\(\)\+\*]")


def resolve_moe_layer_counts(
    num_layers: int,
    moe_layer_freq: Union[int, str],
    first_k_dense_replace: int = 0,
) -> tuple[int, int, list[int]]:
    """Resolve MoE vs Dense layer counts from framework config parameters.

    Parameters
    ----------
    num_layers:
        Total number of transformer layers (``--num-layers``).
    moe_layer_freq:
        ``--moe-layer-freq`` value. Accepts:

        * ``int`` — periodic: an integer *N* means a [0]×(N-1) + [1] cycle.
          *N=1* ⇒ all MoE.  *N=2* ⇒ alternating Dense/MoE.
        * ``str`` — either a bare numeric string (``"3"``) or a Python list
          expression (``"[0]*3+[1]*58"``).  Only list-literal tokens and the
          ``+`` / ``*`` operators are permitted; arbitrary code is rejected.
    first_k_dense_replace:
        ``--first-k-dense-replace`` (MindSpeed-LLM extension).  When > 0 the
        first *K* layers are forced to Dense, regardless of the pattern below
        them.  Default 0 (no override).

    Returns
    -------
    (num_moe_layers, num_dense_layers, pattern_01)
        *pattern_01* is the full-length 0/1 list where 1 = MoE layer.

    Raises
    ------
    ValueError
        If the pattern string is invalid or its length ≠ *num_layers*.
    """
    # ── Normalise MindSpeed sentinel ──
    if isinstance(moe_layer_freq, int) and moe_layer_freq == -1:
        moe_layer_freq = 1

    # ── Step 1: evaluate pattern for the full layer stack ──
    full_pattern = _eval_layer_freq(num_layers, moe_layer_freq)

    # ── Step 2: overlay first-k-dense-replace ──
    if first_k_dense_replace > 0:
        k = min(first_k_dense_replace, num_layers)
        full_pattern[:k] = [0] * k

    num_moe = sum(full_pattern)
    num_dense = num_layers - num_moe
    return num_moe, num_dense, full_pattern


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _eval_layer_freq(num_layers: int, moe_layer_freq) -> list[int]:
    """Convert *moe_layer_freq* into a length-*num_layers* 0/1 list."""

    if isinstance(moe_layer_freq, int):
        cycle = [0] * (moe_layer_freq - 1) + [1]
        q, r = divmod(num_layers, len(cycle))
        return cycle * q + cycle[:r]

    if isinstance(moe_layer_freq, str) and moe_layer_freq.strip().lstrip("-").isdigit():
        return _eval_layer_freq(num_layers, int(moe_layer_freq))

    if isinstance(moe_layer_freq, str) and "[" in moe_layer_freq:
        return _eval_pattern_expr(moe_layer_freq, num_layers)

    raise ValueError(
        f"无法解析 moe_layer_freq: {moe_layer_freq!r} "
        f"(type={type(moe_layer_freq).__name__})"
    )


def _eval_pattern_expr(expr: str, expected_length: int) -> list[int]:
    """Safely evaluate a Python list-expression string.

    Only list-literal characters (digits, commas, brackets, parentheses)
    and ``+`` / ``*`` operators are permitted — matching Megatron-LM's
    ``_eval_pattern`` guard.  This blocks function calls, attribute
    access, import statements, and any other executable code.
    """
    expr = expr.strip()

    if _PATTERN_SAFE_RE.search(expr):
        raise ValueError(
            f"moe_layer_freq 表达式包含非法字符: {expr!r}. "
            f"仅允许数字、逗号、方括号、圆括号、+、*"
        )

    try:
        pattern = eval(expr, {"__builtins__": {}}, {})
    except Exception as exc:
        raise ValueError(f"moe_layer_freq 表达式求值失败: {expr!r}") from exc

    if not isinstance(pattern, list):
        raise ValueError(f"moe_layer_freq 表达式结果应为 list，实际为 {type(pattern).__name__}")

    if len(pattern) != expected_length:
        raise ValueError(
            f"moe_layer_freq pattern 长度 ({len(pattern)}) 与 num_layers "
            f"({expected_length}) 不一致"
        )

    for v in pattern:
        if v not in (0, 1):
            raise ValueError(f"moe_layer_freq pattern 只能包含 0 或 1，发现: {v}")

    return pattern
