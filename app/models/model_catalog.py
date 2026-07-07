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

import logging

import requests

logger = logging.getLogger(__name__)

# ── Builtin dense model fallback table (offline) ──
# Values match public config.json. Used when network fetch fails or for bare
# model names that cannot be resolved to a full org/name repo id.
BUILTIN_DENSE_MODELS: dict[str, dict] = {
    # Reserved for future non-MindSpeed model entries.
    # MindSpeed-sourced models live in MINDSPEED_DENSE_MODELS below.
}

# ── MindSpeed dense model catalog (source: mindspeed) ──
# Parameters extracted from official MindSpeed-LLM examples/mcore + examples/fsdp2
# pretraining scripts. Values represent the canonical training configuration per model.
MINDSPEED_DENSE_MODELS: dict[str, dict] = {
    # ── Qwen3 (MCore) ──
    "Qwen3-0.6B": {
        "num_layers": 28, "d_model": 1024, "num_heads": 16,
        "d_ffn": 3072, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_0point6b_4K_ptd.sh",
    },
    "Qwen3-1.7B": {
        "num_layers": 28, "d_model": 2048, "num_heads": 16,
        "d_ffn": 6144, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh",
    },
    "Qwen3-4B": {
        "num_layers": 36, "d_model": 2560, "num_heads": 32,
        "d_ffn": 9728, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_4b_4K_ptd.sh",
    },
    "Qwen3-8B": {
        "num_layers": 36, "d_model": 4096, "num_heads": 32,
        "d_ffn": 12288, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh",
    },
    "Qwen3-14B": {
        "num_layers": 40, "d_model": 5120, "num_heads": 40,
        "d_ffn": 17408, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh",
    },
    "Qwen3-32B": {
        "num_layers": 64, "d_model": 5120, "num_heads": 64,
        "d_ffn": 25600, "vocab_size": 151936, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen3/pretrain_qwen3_32b_4K_ptd.sh",
    },
    # ── Qwen2.5 (MCore) ──
    "Qwen2.5-72B": {
        "num_layers": 80, "d_model": 8192, "num_heads": 64,
        "d_ffn": 29568, "vocab_size": 152064, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/qwen25/pretrain_qwen25_72b_4k_pack.sh",
    },
    # ── Gemma2 (MCore) ──
    "Gemma2-9B": {
        "num_layers": 42, "d_model": 3584, "num_heads": 16,
        "d_ffn": 14336, "vocab_size": 256000, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/gemma2/pretrain_gemma2_9b_ptd.sh",
    },
    "Gemma2-27B": {
        "num_layers": 46, "d_model": 4608, "num_heads": 32,
        "d_ffn": 36864, "vocab_size": 256000, "num_key_value_heads": 16,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/gemma2/pretrain_gemma2_27b_ptd.sh",
    },
    # ── Llama-2 (MCore) ──
    "Llama-2-13B": {
        "num_layers": 40, "d_model": 5120, "num_heads": 40,
        "d_ffn": 13824, "vocab_size": 32000, "num_key_value_heads": None,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/llama2/pretrain_llama2_13b_ptd.sh",
    },
    # ── Phi3.5 (MCore) ──
    "Phi3.5-Mini": {
        "num_layers": 32, "d_model": 3072, "num_heads": 32,
        "d_ffn": 8192, "vocab_size": 32064, "num_key_value_heads": None,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/phi35/pretrain_phi35_mini_A3_ptd.sh",
    },
    # ── PLM (MCore) ──
    "PLM-1.8B": {
        "num_layers": 32, "d_model": 2048, "num_heads": 16,
        "d_ffn": 8192, "vocab_size": 151936, "num_key_value_heads": 16,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/plm/pretrain_plm_1point8b_ptd.sh",
    },
    # ── Seed-OSS (MCore) ──
    "Seed-OSS-36B": {
        "num_layers": 64, "d_model": 5120, "num_heads": 80,
        "d_ffn": 27648, "vocab_size": 155136, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/seed_oss/pretrain_seed_oss_36b_ptd_2k_A3.sh",
    },
    # ── Magistral (MCore) ──
    "Magistral-Small-24B": {
        "num_layers": 40, "d_model": 5120, "num_heads": 32,
        "d_ffn": 32768, "vocab_size": 131072, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "mindspeed",
        "reference": "https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/magistral/pretrain_magistral_small_24b_A3_ptd.sh",
    },
}

# ── Megatron-LM official dense model catalog (source: megatron) ──
# Parameters extracted from official Megatron-LM examples/ pretrain + RL +
# modelopt scripts. Excludes models already covered by MINDSPEED_DENSE_MODELS.
MEGATRON_DENSE_MODELS: dict[str, dict] = {
    # ── GPT-3 (pretrain) ──
    "GPT-3-175B": {
        "num_layers": 96, "d_model": 12288, "num_heads": 96,
        "d_ffn": 49152, "vocab_size": 50257, "num_key_value_heads": None,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/gpt3/train_gpt3_175b_distributed.sh",
        "description": "Megatron-LM gpt3/train_gpt3_175b_distributed.sh; ffn=4×H default",
    },
    # ── BERT (pretrain) ──
    "BERT-340M": {
        "num_layers": 24, "d_model": 1024, "num_heads": 16,
        "d_ffn": 4096, "vocab_size": 30522, "num_key_value_heads": None,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/bert/train_bert_340m_distributed.sh",
        "description": "Megatron-LM bert/train_bert_340m_distributed.sh; encoder-only, GeLU, LayerNorm",
    },
    # ── T5 (pretrain) ──
    "T5-220M": {
        "num_layers": 12, "d_model": 768, "num_heads": 12,
        "d_ffn": 3072, "vocab_size": 32128, "num_key_value_heads": None,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/t5/train_t5_220m_distributed.sh",
        "description": "Megatron-LM t5/train_t5_220m_distributed.sh; encoder-decoder (12E+12D), ReLU",
    },
    # ── Llama3 (pretrain) ──
    "Llama3-8B": {
        "num_layers": 32, "d_model": 4096, "num_heads": 32,
        "d_ffn": 14336, "vocab_size": 128256, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/train_llama3_8b_h100_fp8.sh",
    },
    # ── Llama3.1 (RL) ──
    "Llama3.1-8B": {
        "num_layers": 32, "d_model": 4096, "num_heads": 32,
        "d_ffn": 14336, "vocab_size": 128256, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/model_configs/llama3p1_8b_instruct.sh",
        "description": "Megatron-LM rl/model_configs/llama3p1_8b_instruct.sh; rope-base=500K",
    },
    # ── Llama3.2 (modelopt) ──
    "Llama3.2-1B": {
        "num_layers": 16, "d_model": 2048, "num_heads": 32,
        "d_ffn": 8192, "vocab_size": 128256, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/conf/meta-llama/Llama-3.2-1B-Instruct.sh",
    },
    # ── Qwen2.5 (modelopt + RL) ──
    "Qwen2.5-0.5B": {
        "num_layers": 24, "d_model": 896, "num_heads": 14,
        "d_ffn": 4864, "vocab_size": 151936, "num_key_value_heads": 2,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/conf/Qwen/Qwen2.5-0.5B-Instruct.sh",
    },
    "Qwen2.5-3B": {
        "num_layers": 36, "d_model": 2048, "num_heads": 16,
        "d_ffn": 11008, "vocab_size": 151936, "num_key_value_heads": 2,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/model_configs/qwen_2p5_3b.sh",
    },
    "Qwen2.5-7B": {
        "num_layers": 28, "d_model": 3584, "num_heads": 28,
        "d_ffn": 18944, "vocab_size": 152064, "num_key_value_heads": 4,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/conf/Qwen/Qwen2.5-7B-Instruct.sh",
    },
    "Qwen2.5-32B": {
        "num_layers": 64, "d_model": 5120, "num_heads": 40,
        "d_ffn": 27648, "vocab_size": 152064, "num_key_value_heads": 8,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/model_configs/qwen_2p5_32b.sh",
    },
    # ── Qwen2.5-Math (RL) ──
    "Qwen2.5-Math-7B": {
        "num_layers": 28, "d_model": 3584, "num_heads": 28,
        "d_ffn": 18944, "vocab_size": 152064, "num_key_value_heads": 4,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/model_configs/qwen_2p5_math_7b.sh",
        "description": "Megatron-LM rl/model_configs/qwen_2p5_math_7b.sh; rope-base=10K",
    },
    # ── DeepSeek-R1-Distill-Qwen (RL) ──
    "DeepSeek-R1-Distill-Qwen-7B": {
        "num_layers": 28, "d_model": 3584, "num_heads": 28,
        "d_ffn": 18944, "vocab_size": 152064, "num_key_value_heads": 4,
        "model_type": "dense", "_source": "megatron",
        "reference": "https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/model_configs/qwen_2p5_distill_7b.sh",
        "description": "Megatron-LM rl/model_configs/qwen_2p5_distill_7b.sh; rope-base=10K",
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


def _normalize_hf_config(hf_config: dict, source: str, model_id: str = "") -> dict | None:
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
    # Set reference URL pointing to the model page
    if model_id:
        if source == "huggingface":
            out["reference"] = f"https://huggingface.co/{model_id}"
        elif source == "modelscope":
            out["reference"] = f"https://modelscope.cn/models/{model_id}"
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


def _builtin_match(model_name: str) -> dict | None:
    """Fuzzy match a (possibly bare) model name against offline catalog tables."""
    norm = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    # MindSpeed catalog (primary)
    for name, cfg in MINDSPEED_DENSE_MODELS.items():
        if name.lower().replace("-", "").replace("_", "").replace(" ", "") == norm:
            return {**cfg}
    # Megatron-LM official catalog
    for name, cfg in MEGATRON_DENSE_MODELS.items():
        if name.lower().replace("-", "").replace("_", "").replace(" ", "") == norm:
            return {**cfg}
    # BUILTIN_DENSE_MODELS (reserved)
    for name, cfg in BUILTIN_DENSE_MODELS.items():
        if name.lower().replace("-", "").replace("_", "").replace(" ", "") == norm:
            return {**cfg}
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
        normalized = _normalize_hf_config(hf_raw, "huggingface", repo_id)
        if normalized:
            return normalized
    ms_raw = _fetch_modelscope_config(repo_id)
    if ms_raw:
        return _normalize_hf_config(ms_raw, "modelscope", repo_id)
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
    """Persist a resolved config to PostgreSQL (best-effort).

    All sources (megatron / huggingface / modelscope / mindspeed / manual) are
    persisted uniformly. Failures are logged and swallowed so a DB outage never
    breaks resolution.
    """
    try:
        from app.dao import upsert_model_catalog
        upsert_model_catalog(model_name, resolved)
    except Exception as e:
        logger.warning(f"[model_catalog] pg upsert failed for {model_name}: {e}")


def resolve_model_config(model_name: str) -> dict | None:
    """
    Layered resolution:
      1. PostgreSQL catalog — source priority: mindspeed > megatron > huggingface > modelscope
      2. remote fetch: full org/name repo id → direct (HF → ModelScope);
         bare name → HF search resolves the canonical repo, then fetch

    Builtin models (MINDSPEED_DENSE_MODELS, MEGATRON_DENSE_MODELS) are seeded
    into PG at startup via init_db(). The DB query's source priority ensures
    official entries always outrank previously cached remote fetches.

    On a remote hit the result is upserted into PG so future lookups hit PG.
    Returns dict with num_layers/d_model/num_heads/d_ffn/vocab_size/model_type/
    _source, or None if not found. Sparse models are returned with
    model_type="sparse" so the caller can refuse them.
    """
    if not model_name:
        return None

    # 1. PostgreSQL catalog (source-prioritized: mindspeed > megatron > hf > ms)
    try:
        from app.dao import get_model_catalog_entry
        pg = get_model_catalog_entry(model_name)
        if pg:
            logger.info(f"[model_catalog] pg hit: {model_name} (source={pg.get('_source')})")
            return pg
    except Exception as e:
        logger.warning(f"[model_catalog] pg lookup failed for {model_name}: {e}")

    # 2. Remote fetch — only when DB has no match
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

    if not resolved:
        return None

    # Persist so future lookups hit PG
    _persist_resolved(model_name, resolved)
    return resolved
