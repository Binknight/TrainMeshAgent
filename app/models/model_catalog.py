"""
Model catalog & resolver — fetch model architecture config (L/H/A/dff/V) from
HuggingFace / ModelScope config.json, with local JSON cache and a builtin
offline fallback table.

Only dense (non-MoE) models are supported in v1. Sparse/MoE models are detected
and surfaced as model_type="sparse" so the caller can refuse them.

Returned field names mirror TrainingModelConfig (app/models/schemas.py) so the
result can be reused directly:
  num_layers, d_model, num_heads, d_ffn, vocab_size (+ model_type, _source)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Local JSON cache (gitignored) ──
_CACHE_DIR = Path(__file__).parent / ".model_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Builtin dense model fallback table (offline) ──
# Values match public config.json. Used when network fetch fails or for bare
# model names that cannot be resolved to a full org/name repo id.
BUILTIN_DENSE_MODELS: dict[str, dict] = {
    "Qwen3-32B": {
        "num_layers": 64, "d_model": 4096, "num_heads": 32,
        "d_ffn": 14336, "vocab_size": 32000, "model_type": "dense",
    },
    "Qwen2.5-7B": {
        "num_layers": 28, "d_model": 3584, "num_heads": 28,
        "d_ffn": 18944, "vocab_size": 152064, "model_type": "dense",
    },
    "Qwen2.5-14B": {
        "num_layers": 48, "d_model": 5120, "num_heads": 40,
        "d_ffn": 13824, "vocab_size": 152064, "model_type": "dense",
    },
    "Qwen2.5-32B": {
        "num_layers": 64, "d_model": 5120, "num_heads": 40,
        "d_ffn": 27648, "vocab_size": 152064, "model_type": "dense",
    },
    "Qwen2.5-72B": {
        "num_layers": 80, "d_model": 8192, "num_heads": 64,
        "d_ffn": 29568, "vocab_size": 152064, "model_type": "dense",
    },
    "LLaMA-7B": {
        "num_layers": 32, "d_model": 4096, "num_heads": 32,
        "d_ffn": 11008, "vocab_size": 32000, "model_type": "dense",
    },
    "LLaMA-13B": {
        "num_layers": 40, "d_model": 5120, "num_heads": 40,
        "d_ffn": 13824, "vocab_size": 32000, "model_type": "dense",
    },
    "LLaMA-70B": {
        "num_layers": 80, "d_model": 8192, "num_heads": 64,
        "d_ffn": 28672, "vocab_size": 32000, "model_type": "dense",
    },
}

# HuggingFace config.json field → internal field (mirrors TrainingModelConfig)
_HF_FIELD_MAP = {
    "num_hidden_layers": "num_layers",
    "hidden_size": "d_model",
    "num_attention_heads": "num_heads",
    "intermediate_size": "d_ffn",
    "vocab_size": "vocab_size",
}

_HTTP_TIMEOUT = 10


# Config fields that only appear in MoE / sparse models (any vendor).
# Dense models never have these, so mere presence => sparse. This catches
# Mixtral (num_experts), DeepSeek-V2/V3 (n_routed_experts, moe_layer_freq,
# first_k_dense_replace, ...), Qwen-MoE, etc. — which the num_experts>1 /
# architectures-substring checks alone miss.
_MOE_INDICATOR_FIELDS = (
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "moe_layer_freq",
    "moe_intermediate_size",
    "expert_intermediate_size",
    "first_k_dense_replace",
    "moe_num_experts",
    "decoder_sparse_step",
)


def _is_sparse(hf_config: dict) -> bool:
    """Detect MoE / sparse models across vendors (Mixtral, DeepSeek-V2/V3, Qwen-MoE, ...)."""
    # 1. Any MoE-specific config field present => sparse (num_experts needs >1)
    if "num_experts" in hf_config:
        try:
            if int(hf_config.get("num_experts", 1) or 1) > 1:
                return True
        except (TypeError, ValueError):
            pass
    for field in _MOE_INDICATOR_FIELDS:
        if field in hf_config:
            return True
    # 2. architectures substring (Mixtral, *MoE*)
    archs = hf_config.get("architectures") or []
    if isinstance(archs, list):
        for a in archs:
            if isinstance(a, str) and ("MoE" in a or "Mixtral" in a):
                return True
    # 3. model_type substring
    if "moe" in (hf_config.get("model_type") or "").lower():
        return True
    return False


def _normalize_hf_config(hf_config: dict, source: str) -> dict | None:
    """Map a raw HF/ModelScope config.json to the internal arch dict."""
    out: dict = {}
    for hf_key, our_key in _HF_FIELD_MAP.items():
        if hf_key in hf_config:
            out[our_key] = hf_config[hf_key]
    # Required architecture fields
    if "num_layers" not in out or "d_model" not in out or "num_heads" not in out:
        return None
    # dff fallback: n_inner (GPT-2 style) → 4 * d_model
    if "d_ffn" not in out:
        n_inner = hf_config.get("n_inner")
        out["d_ffn"] = n_inner if n_inner else 4 * int(out["d_model"])
    out.setdefault("vocab_size", 32000)
    # Capture GQA kv-head count for future TP-comm accuracy (unused in v1)
    kv = hf_config.get("num_key_value_heads")
    if kv:
        out["num_key_value_heads"] = kv
    out["model_type"] = "sparse" if _is_sparse(hf_config) else "dense"
    out["_source"] = source
    return out


def _fetch_hf_config(model_id: str) -> dict | None:
    """Fetch raw config.json from HuggingFace Hub (public models, no auth)."""
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        logger.info(f"[model_catalog] HF {model_id} status={resp.status_code}")
    except Exception as e:
        logger.warning(f"[model_catalog] HF fetch failed for {model_id}: {e}")
    return None


def _fetch_modelscope_config(model_id: str) -> dict | None:
    """Fetch raw config.json from ModelScope (file-download API)."""
    url = (
        "https://modelscope.cn/api/v1/models/"
        f"{model_id}/repo?Revision=master&FilePath=config.json"
    )
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            # File endpoint returns raw config; validate it's a real model config
            if isinstance(data, dict) and "num_hidden_layers" in data:
                return data
    except Exception as e:
        logger.warning(f"[model_catalog] ModelScope fetch failed for {model_id}: {e}")
    return None


def _cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__").replace("\\", "__")
    return _CACHE_DIR / f"{safe}.json"


def _builtin_match(model_name: str) -> dict | None:
    """Fuzzy match a (possibly bare) model name against the builtin table."""
    norm = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, cfg in BUILTIN_DENSE_MODELS.items():
        if key.lower().replace("-", "").replace("_", "").replace(" ", "") == norm:
            return {**cfg, "_source": "builtin"}
    return None


# Official-ish vendor orgs on HuggingFace — used to prefer canonical repos when
# a bare-name search returns both official and community/quantized variants.
_OFFICIAL_ORGS = {
    "deepseek-ai", "qwen", "qwenlm", "meta-llama", "mistralai",
    "google", "allenai", "bigscience", "eleutherai", "tiiuae",
    "microsoft", "nvidia", "baai", "thudm", "internlm", "alibaba-pai",
}


def _normalize_for_match(name: str) -> str:
    """Normalize for fuzzy name comparison: lowercase, strip - _ and spaces."""
    return (name or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def _remote_fetch(repo_id: str) -> dict | None:
    """Fetch + normalize config from HuggingFace then ModelScope."""
    hf_raw = _fetch_hf_config(repo_id)
    if hf_raw:
        normalized = _normalize_hf_config(hf_raw, "huggingface")
        if normalized:
            return normalized
    ms_raw = _fetch_modelscope_config(repo_id)
    if ms_raw:
        return _normalize_hf_config(ms_raw, "modelscope")
    return None


def _search_hf_model(bare_name: str) -> str | None:
    """Search HuggingFace for a bare model name; return the best canonical repo id.

    Prefers an exact name-segment match from an official vendor org, falling back
    to the highest-scoring public/non-gated result. Returns 'org/name' or None.
    """
    try:
        resp = requests.get(
            "https://huggingface.co/api/models",
            params={"search": bare_name, "limit": 20},
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        items = resp.json()
    except Exception as e:
        logger.warning(f"[model_catalog] HF search failed for {bare_name}: {e}")
        return None

    target = _normalize_for_match(bare_name)
    best_id = None
    best_score = -1
    for it in items:
        repo_id = it.get("id", "")
        if not repo_id or "/" not in repo_id:
            continue
        if it.get("private"):
            continue
        org, _, name = repo_id.partition("/")
        name_norm = _normalize_for_match(name)
        score = 0
        if name_norm == target:
            score += 100
        elif target and target in name_norm:
            score += 30
        if org.lower() in _OFFICIAL_ORGS:
            score += 20
        if it.get("gated") in (None, False, "false"):
            score += 5
        if score > best_score:
            best_score = score
            best_id = repo_id
    return best_id if best_score > 0 else None


def _persist_resolved(model_name: str, resolved: dict) -> None:
    """Persist a resolved config to disk cache + PostgreSQL (best-effort).

    Builtin-sourced configs are seeded into PG at boot, so only the disk cache
    is refreshed for them. Failures are logged and swallowed so a DB/disk
    outage never breaks resolution.
    """
    # disk cache (local fallback when PG is unavailable)
    try:
        with open(_cache_path(model_name), "w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[model_catalog] cache write failed: {e}")
    # PG upsert (shared store) — skip builtin (seeded at boot)
    if resolved.get("_source") == "builtin":
        return
    try:
        from app.dao import upsert_model_catalog
        upsert_model_catalog(model_name, resolved)
    except Exception as e:
        logger.warning(f"[model_catalog] pg upsert failed for {model_name}: {e}")


def resolve_model_config(model_name: str) -> dict | None:
    """
    Layered resolution:
      1. PostgreSQL catalog (curated + previously fetched; shared across instances)
      2. local JSON disk cache (per-instance fallback)
      3. remote fetch: full org/name repo id → direct (HF → ModelScope);
         bare name → HF search resolves the canonical repo, then fetch
      4. builtin fuzzy fallback (offline, handles bare names like 'Qwen3-32B')

    On a remote hit the result is upserted into PG so future lookups hit PG.
    Returns dict with num_layers/d_model/num_heads/d_ffn/vocab_size/model_type/
    _source, or None if not found. Sparse models are returned with
    model_type="sparse" so the caller can refuse them.
    """
    if not model_name:
        return None

    # 1. PostgreSQL catalog
    try:
        from app.dao import get_model_catalog_entry
        pg = get_model_catalog_entry(model_name)
        if pg:
            logger.info(f"[model_catalog] pg hit: {model_name}")
            return pg
    except Exception as e:
        logger.warning(f"[model_catalog] pg lookup failed for {model_name}: {e}")

    # 2. Local disk cache
    cache_file = _cache_path(model_name)
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                logger.info(f"[model_catalog] cache hit: {model_name}")
                return json.load(f)
        except Exception:
            pass  # corrupt cache → fall through

    # 3. Remote fetch (authoritative):
    #    - full org/name repo id → direct fetch
    #    - bare name → HF search resolves the canonical repo, then fetch
    resolved = None
    if "/" in model_name:
        resolved = _remote_fetch(model_name)
    else:
        repo_id = _search_hf_model(model_name)
        if repo_id:
            logger.info(
                f"[model_catalog] search resolved {model_name!r} -> {repo_id!r}"
            )
            resolved = _remote_fetch(repo_id)

    # 4. Builtin fuzzy fallback (offline / remote miss)
    if not resolved:
        resolved = _builtin_match(model_name)

    if not resolved:
        return None

    # Persist so future lookups hit PG (and refresh disk cache)
    _persist_resolved(model_name, resolved)
    return resolved
