"""Canonical global-rank layout for a training mesh (single source of truth).

The rank numbering is calibrated to the **simulation system (MCP Server)**,
whose dimension order is::

    TP - DP - PP        (fastest-varying  ->  slowest-varying)

so that::

    global_rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank

and inversely::

    tp_rank = global_rank % tp
    dp_rank = (global_rank // tp) % dp
    pp_rank = global_rank // (tp * dp)

Why this matters
----------------
The previous layout was ``TP-PP-DP``
(``global_rank = dp_rank * (tp * pp) + pp_rank * tp + tp_rank``).
Because ``pp_rank`` was derived locally with ``(rank // tp) % pp`` while the
simulation system derives it with ``rank // (tp * dp)``, the *same*
``global_rank`` was attributed to a different pipeline stage on the two sides.
PP communication is the only metric that differs per pipeline stage, so
per-stage PP comparison (first / middle / last) silently compared the metrics
of different stages and reported "PP not equivalent".

Rules
-----
* Every place that converts between ``global_rank`` and ``(dp, tp, pp)`` ranks
  — Python or JavaScript — must use these formulas (or the JS mirrors
  ``meshRankOf`` / ``meshDecomposeRank`` in ``static/topo-renderer.js``).
* Changing the layout changes **both** the original and the equivalent
  topology, because both are generated from these helpers.
* ``total_nodes`` is unchanged: ``dp * tp * pp``.
"""

from __future__ import annotations

from typing import Tuple

#: Dimension order, fastest-varying first. Must match the simulation system.
RANK_ORDER = "TP-DP-PP"


def _sizes(dp: int, tp: int) -> Tuple[int, int]:
    """Guard against zero/negative parallelism degrees."""
    return max(1, int(dp)), max(1, int(tp))


def rank_of(dp: int, tp: int, *, dp_rank: int, tp_rank: int, pp_rank: int) -> int:
    """Compose a ``global_rank`` from ``(dp_rank, tp_rank, pp_rank)``."""
    dp, tp = _sizes(dp, tp)
    return int(pp_rank) * (tp * dp) + int(dp_rank) * tp + int(tp_rank)


def decompose(global_rank: int, dp: int, tp: int, pp: int) -> Tuple[int, int, int]:
    """Split ``global_rank`` into ``(dp_rank, tp_rank, pp_rank)``.

    ``pp`` is accepted for signature symmetry/validation only; the decomposition
    is fully determined by ``tp`` and ``dp``.
    """
    dp, tp = _sizes(dp, tp)
    g = int(global_rank)
    tp_rank = g % tp
    dp_rank = (g // tp) % dp
    pp_rank = g // (tp * dp)
    return dp_rank, tp_rank, pp_rank


def pp_rank_of(global_rank: int, dp: int, tp: int) -> int:
    """Pipeline-stage index of ``global_rank`` (the hot path for PP metrics)."""
    dp, tp = _sizes(dp, tp)
    return int(global_rank) // (tp * dp)


def dp_rank_of(global_rank: int, dp: int, tp: int) -> int:
    """Data-parallel replica index of ``global_rank``."""
    dp, tp = _sizes(dp, tp)
    return (int(global_rank) // tp) % dp


def tp_rank_of(global_rank: int, tp: int) -> int:
    """Tensor-parallel index of ``global_rank`` (innermost dimension)."""
    _, tp = _sizes(1, tp)
    return int(global_rank) % tp


def ranks_per_pp(dp: int, tp: int) -> int:
    """Number of ranks belonging to one pipeline stage (one PP replica)."""
    dp, tp = _sizes(dp, tp)
    return tp * dp


