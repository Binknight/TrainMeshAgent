"""Model catalog management REST endpoints (admin CRUD + remote import).

No auth — matches the existing /api surface (CORS open). Add auth if exposed
beyond a trusted internal network.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.dao import (
    delete_model_catalog_entry,
    get_model_catalog_entry,
    list_model_catalog,
    seed_model_catalog_builtin,
    upsert_model_catalog,
)
from app.models.model_catalog import (
    MINDSPEED_DENSE_MODELS, MEGATRON_DENSE_MODELS,
    MINDSPEED_MOE_MODELS, MEGATRON_MOE_MODELS,
    resolve_model_config,
)

logger = logging.getLogger(__name__)

model_catalog_bp = Blueprint("model_catalog", __name__, url_prefix="/api/model-catalog")

# Architecture fields a manually-entered dense model must provide.
_REQUIRED_ARCH_FIELDS = ("num_layers", "d_model", "num_heads", "d_ffn", "vocab_size")


@model_catalog_bp.route("", methods=["GET"])
def list_or_get():
    """List all entries, or fetch one by name.

    GET /api/model-catalog            → all entries
    GET /api/model-catalog?name=X     → single entry (name may contain '/')
    """
    name = request.args.get("name")
    if name:
        entry = get_model_catalog_entry(name)
        if not entry:
            return {"error": f"未找到模型 {name}"}, 404
        return jsonify(entry)
    return jsonify(list_model_catalog())


@model_catalog_bp.route("", methods=["POST"])
def upsert():
    """Create or update a model entry (idempotent upsert).

    Body: {model_name, num_layers, d_model, num_heads, d_ffn, vocab_size,
           model_type?, num_key_value_heads?, description?, source?, reference?}
    """
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    if not model_name:
        return {"error": "model_name 必填"}, 400
    missing = [f for f in _REQUIRED_ARCH_FIELDS if data.get(f) is None]
    if missing:
        return {"error": f"缺少必填架构字段: {', '.join(missing)}"}, 400
    try:
        cfg = {
            "model_type": data.get("model_type") or "dense",
            "num_layers": int(data["num_layers"]),
            "d_model": int(data["d_model"]),
            "num_heads": int(data["num_heads"]),
            "d_ffn": int(data["d_ffn"]),
            "vocab_size": int(data["vocab_size"]),
            "num_key_value_heads": data.get("num_key_value_heads"),
            "_source": data.get("source") or "manual",
            "reference": data.get("reference"),
        }
    except (TypeError, ValueError) as e:
        return {"error": f"架构字段必须为整数: {e}"}, 400
    upsert_model_catalog(
        model_name, cfg, description=data.get("description")
    )
    logger.info(f"[model_catalog_api] upserted {model_name}")
    return jsonify(get_model_catalog_entry(model_name)), 200


@model_catalog_bp.route("", methods=["DELETE"])
def delete():
    """Delete an entry by name. DELETE /api/model-catalog?name=X"""
    name = request.args.get("name")
    if not name:
        return {"error": "name 查询参数必填"}, 400
    if not delete_model_catalog_entry(name):
        return {"error": f"未找到模型 {name}"}, 404
    return jsonify({"deleted": name})


@model_catalog_bp.route("/seed", methods=["POST"])
def reseed():
    """Re-seed the model catalog (idempotent upsert)."""
    count1 = seed_model_catalog_builtin(MINDSPEED_DENSE_MODELS)
    count2 = seed_model_catalog_builtin(MEGATRON_DENSE_MODELS)
    count3 = seed_model_catalog_builtin(MINDSPEED_MOE_MODELS)
    count4 = seed_model_catalog_builtin(MEGATRON_MOE_MODELS)
    logger.info(f"[model_catalog_api] re-seeded {count1} mindspeed dense + {count2} megatron dense + {count3} mindspeed moe + {count4} megatron moe models")
    return jsonify({
        "mindspeed_dense": count1, "megatron_dense": count2,
        "mindspeed_moe": count3, "megatron_moe": count4,
        "total": count1 + count2 + count3 + count4,
    })


@model_catalog_bp.route("/fetch", methods=["POST"])
def fetch_remote():
    """Import a model from HuggingFace/ModelScope by name or repo-id.

    Resolves via the full pipeline (PG → cache → remote → builtin) and
    persists to PG, so subsequent lookups hit PG. Refuses sparse models.
    Body: {model_name}
    """
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    if not model_name:
        return {"error": "model_name 必填"}, 400
    resolved = resolve_model_config(model_name)
    if not resolved:
        return {"error": f"未找到模型 {model_name} 的配置"}, 404
    return jsonify(get_model_catalog_entry(model_name)), 201
