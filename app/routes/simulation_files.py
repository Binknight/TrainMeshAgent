"""仿真产物文件下载（Flask 侧）：列表 / 单文件 / ZIP 打包。

数据来源是 MCP Server 的 REST 扩展（``/tasks/{task_id}/files*``，见
``mcp_server/services/task_files.py``）：**Flask 不直接读 workspace**，只负责
把会话两侧的 ``original_task_id`` / ``equivalent_task_id`` 绑定到请求的 ``side``，
再代理 MCP 的流式响应。这样：

- 任务目录是否已被磁盘清理、相对路径是否越权，都以 MCP 侧校验为准；
- 大文件（单个 CSV / 整个 results ZIP）不会被 Flask 读进内存；
- 下载入口与「仿真验证」tab 绑定：无 task_id 的会话无法白拿任意任务文件。

接口（浏览器 / 前端调用）：
- ``GET /api/session/<sid>/simulation-files``
  返回两侧任务状态、文件清单与分类分组（供下载面板渲染）。
- ``GET /api/session/<sid>/simulation-files/download?side=original&path=...``
  单文件流式下载（side 必须是 original/equivalent，且该侧有 task_id）。
- ``GET /api/session/<sid>/simulation-files/bundle?side=original|equivalent|both&category=...``
  ZIP 打包下载；``side=both`` 时由 Flask 把两侧文件合并进一个 ZIP（按
  ``原始组网/``、``等效组网/`` 分目录）。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import zipfile
from urllib.parse import quote

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from app.agent.session import session_manager
from app.mcp.client import mcp_client

logger = logging.getLogger(__name__)

files_bp = Blueprint("simulation_files", __name__, url_prefix="/api/session")

SIDES = ("original", "equivalent")
SIDE_LABELS = {"original": "原始组网", "equivalent": "等效组网"}
# 与 mcp_server/services/task_files.py 的 CATEGORY_LABELS 保持一致
CATEGORY_LABELS = {
    "traffic_detail": "流量明细",
    "operator_timeline": "算子时间线",
    "statistic_summary": "统计汇总",
    "script": "训练脚本",
    "logs": "运行日志",
    "other": "其他文件",
}
STATUS_LABELS = {
    "not_started": "未开始",
    "submitted": "已提交",
    "queued": "排队中",
    "running": "仿真中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "unavailable": "读取失败",
    "unknown": "未知",
}
# 合并两侧 ZIP 的上限（与 MCP 侧 files_bundle_max_bytes 同量级；防止一次请求
# 把磁盘/网络打满）。超限返回 413，提示用户按分类/单侧下载。
MAX_BUNDLE_BYTES = int(os.getenv("SIM_FILES_MAX_BUNDLE_BYTES", str(2 * 1024**3)))
MAX_BUNDLE_FILES = int(os.getenv("SIM_FILES_MAX_FILES", "20000"))
_CHUNK = 64 * 1024


class _BundleBuildError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _get_session(session_id: str):
    return session_manager.get_session(session_id)


def _session_or_error(session_id: str):
    session = _get_session(session_id)
    if session is None:
        return None, (jsonify({"error": "session_not_found", "message": "会话不存在"}), 404)
    return session, None


def _side_task_id(session, side: str) -> str:
    if side == "original":
        return getattr(session, "original_task_id", "") or ""
    if side == "equivalent":
        return getattr(session, "equivalent_task_id", "") or ""
    return ""


def _attachment_header(filename: str) -> str:
    """兼容中文文件名的 Content-Disposition（ASCII 回退 + RFC 5987）。"""
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "download"
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def _normalize_side(value: str) -> str | None:
    value = (value or "").strip().lower()
    return value if value in (*SIDES, "both") else None


def _normalize_category(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if value not in CATEGORY_LABELS:
        return None
    return value


def _side_summary(session, side: str) -> dict:
    """组装一侧的下载面板数据（文件列表 + 状态 + 错误说明）。"""
    task_id = _side_task_id(session, side)
    payload: dict = {
        "side": side,
        "label": SIDE_LABELS[side],
        "task_id": task_id,
        "available": False,
        "status": "not_started",
        "status_label": STATUS_LABELS["not_started"],
        "files": [],
        "groups": [],
        "total_files": 0,
        "total_size_bytes": 0,
        "truncated": False,
        "message": None,
        "error": None,
    }
    if not task_id:
        payload["error"] = "尚未提交该侧仿真任务"
        return payload

    data = mcp_client.list_task_files(task_id)
    if not isinstance(data, dict) or data.get("status") == "unavailable" or data.get("error"):
        payload["status"] = "unavailable"
        payload["status_label"] = STATUS_LABELS["unavailable"]
        payload["error"] = str(
            data.get("error") if isinstance(data, dict) else "无法读取仿真文件列表"
        )
        return payload

    status = str(data.get("status") or "unknown")
    files = data.get("files") if isinstance(data.get("files"), list) else []
    groups = data.get("groups") if isinstance(data.get("groups"), list) else []
    payload.update(
        available=True,
        status=status,
        status_label=STATUS_LABELS.get(status, status),
        files=files,
        groups=groups,
        total_files=int(data.get("total_files") or len(files)),
        total_size_bytes=int(data.get("total_size_bytes") or 0),
        truncated=bool(data.get("truncated")),
        message=data.get("message"),
    )
    if not files:
        payload["message"] = payload["message"] or "暂无文件（仿真尚未产出或已被清理）"
    return payload


@files_bp.route("/<session_id>/simulation-files", methods=["GET"])
def list_simulation_files(session_id: str):
    """返回会话两侧任务的文件清单与状态（下载面板数据源）。"""
    session, error = _session_or_error(session_id)
    if error:
        return error
    sides = {side: _side_summary(session, side) for side in SIDES}
    return jsonify(
        {
            "session_id": session_id,
            "session_step": getattr(session, "step", None),
            "sides": sides,
            "categories": [
                {"category": key, "label": label}
                for key, label in CATEGORY_LABELS.items()
            ],
            "limits": {
                "max_bundle_bytes": MAX_BUNDLE_BYTES,
                "max_bundle_files": MAX_BUNDLE_FILES,
            },
            "generated_at": time.time(),
        }
    )


def _upstream_error(resp: requests.Response, side: str):
    """把 MCP 的非 200 响应翻译成 Flask JSON 错误。

    400/404/413 属于调用方可理解的错误（路径非法 / 文件或任务不存在 / 打包过大），
    原样透传状态码；其余（MCP 内部错误、网络层 5xx）统一 502。
    """
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("error") or body.get("message") or ""
        else:
            detail = str(body)
    except ValueError:
        detail = (resp.text or "").strip()[:200]
    pass_through = {400: "文件路径不合法", 404: "任务目录或文件不存在（可能已被磁盘清理）"}
    if resp.status_code in pass_through:
        status = resp.status_code
        message = pass_through[status]
    elif resp.status_code == 413:
        status = 413
        message = "打包体积或文件数超出上限，请按分类或单侧下载"
    else:
        status = 502
        message = detail or f"读取仿真文件失败（HTTP {resp.status_code}）"
    logger.warning(
        "simulation files: side=%s upstream HTTP %s: %s",
        side,
        resp.status_code,
        detail,
    )
    return jsonify({"error": "simulation_file_unavailable", "message": message}), status


@files_bp.route("/<session_id>/simulation-files/download", methods=["GET"])
def download_simulation_file(session_id: str):
    """代理下载单个仿真产物文件（流式）。"""
    session, error = _session_or_error(session_id)
    if error:
        return error
    side = _normalize_side(request.args.get("side", ""))
    if side not in SIDES:
        return jsonify({"error": "invalid_side", "message": "side 必须为 original 或 equivalent"}), 400
    rel_path = (request.args.get("path") or "").strip()
    if not rel_path:
        return jsonify({"error": "invalid_path", "message": "缺少文件路径 path"}), 400
    task_id = _side_task_id(session, side)
    if not task_id:
        return jsonify({"error": "task_not_found", "message": "该侧尚未提交仿真任务"}), 404

    try:
        upstream = mcp_client.open_task_file(task_id, rel_path)
    except requests.RequestException as exc:
        logger.warning("simulation file download proxy failed: %s", exc)
        return jsonify({"error": "mcp_unavailable", "message": "无法连接仿真服务，请稍后重试"}), 503
    if upstream.status_code != 200:
        result = _upstream_error(upstream, side)
        upstream.close()
        return result

    filename = os.path.basename(rel_path.replace("\\", "/")) or "simulation_file"

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=_CHUNK):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    resp = Response(stream_with_context(generate()))
    resp.headers["Content-Type"] = (
        upstream.headers.get("Content-Type") or "application/octet-stream"
    )
    content_length = upstream.headers.get("Content-Length")
    if content_length:
        resp.headers["Content-Length"] = content_length
    resp.headers["Content-Disposition"] = _attachment_header(filename)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@files_bp.route("/<session_id>/simulation-files/bundle", methods=["GET"])
def download_simulation_bundle(session_id: str):
    """按侧（original / equivalent）或双侧合并，下载 ZIP。"""
    session, error = _session_or_error(session_id)
    if error:
        return error
    side = _normalize_side(request.args.get("side", "both"))
    if side is None:
        return jsonify({"error": "invalid_side", "message": "side 必须为 original/equivalent/both"}), 400
    category = _normalize_category(request.args.get("category"))
    if request.args.get("category") and category is None:
        return jsonify({"error": "invalid_category", "message": "未知的文件分类"}), 400

    if side == "both":
        return _combined_bundle_response(session, session_id, category)

    task_id = _side_task_id(session, side)
    if not task_id:
        return jsonify({"error": "task_not_found", "message": "该侧尚未提交仿真任务"}), 404
    try:
        upstream = mcp_client.open_task_bundle(task_id, category)
    except requests.RequestException as exc:
        logger.warning("simulation bundle proxy failed: %s", exc)
        return jsonify({"error": "mcp_unavailable", "message": "无法连接仿真服务，请稍后重试"}), 503
    if upstream.status_code != 200:
        result = _upstream_error(upstream, side)
        upstream.close()
        return result

    filename = f"simulation_files_{side}{('_' + category) if category else ''}.zip"

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=_CHUNK):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    resp = Response(stream_with_context(generate()))
    resp.headers["Content-Type"] = "application/zip"
    content_length = upstream.headers.get("Content-Length")
    if content_length:
        resp.headers["Content-Length"] = content_length
    resp.headers["Content-Disposition"] = _attachment_header(filename)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


def _build_combined_bundle(session, session_id: str, category: str | None):
    """把两侧任务文件合并进一个临时 ZIP，返回 (临时路径, 下载文件名)。

    以流式方式逐个从 MCP 拉取文件写入 ZIP，避免把 results/ 整目录读进内存；
    任何一侧失败都中止并清理临时文件（不返回半截 ZIP）。
    """
    sides_data: list[tuple[str, str, list[dict]]] = []
    total_files = 0
    total_bytes = 0
    for side in SIDES:
        task_id = _side_task_id(session, side)
        if not task_id:
            raise _BundleBuildError(404, f"「{SIDE_LABELS[side]}」尚未提交仿真任务")
        data = mcp_client.list_task_files(task_id)
        if not isinstance(data, dict) or data.get("status") == "unavailable" or data.get("error"):
            detail = data.get("error") if isinstance(data, dict) else None
            raise _BundleBuildError(502, f"「{SIDE_LABELS[side]}」文件列表读取失败：{detail or '仿真服务不可用'}")
        files = data.get("files") if isinstance(data.get("files"), list) else []
        if category:
            files = [f for f in files if f.get("category") == category]
        sides_data.append((SIDE_LABELS[side], task_id, files))
        total_files += len(files)
        total_bytes += sum(int(f.get("size") or 0) for f in files)

    if total_files == 0:
        raise _BundleBuildError(404, "没有可打包的文件（仿真尚未产出或已被清理）")
    if total_files > MAX_BUNDLE_FILES:
        raise _BundleBuildError(413, f"文件过多（{total_files} > {MAX_BUNDLE_FILES}），请按分类下载")
    if MAX_BUNDLE_BYTES > 0 and total_bytes > MAX_BUNDLE_BYTES:
        raise _BundleBuildError(413, "打包体积超出上限，请按分类或单侧下载")

    fd, tmp_name = tempfile.mkstemp(prefix="sim_files_", suffix=".zip")
    os.close(fd)
    tmp_path = tmp_name
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for label, task_id, files in sides_data:
                for file_info in files:
                    rel_path = file_info.get("path") or ""
                    upstream = mcp_client.open_task_file(task_id, rel_path)
                    try:
                        if upstream.status_code != 200:
                            raise _BundleBuildError(
                                502,
                                f"读取文件失败：{label}/{rel_path}（HTTP {upstream.status_code}）",
                            )
                        arcname = f"{label}/{rel_path}"
                        with zf.open(arcname, "w") as dest:
                            for chunk in upstream.iter_content(chunk_size=_CHUNK):
                                if chunk:
                                    dest.write(chunk)
                    finally:
                        upstream.close()
    except _BundleBuildError:
        _remove_quietly(tmp_path)
        raise
    except Exception as exc:  # noqa: BLE001 — 网络/磁盘错误统一转 502
        _remove_quietly(tmp_path)
        logger.exception("combined simulation bundle build failed: %s", exc)
        raise _BundleBuildError(502, f"打包失败：{exc}") from exc

    suffix = f"_{category}" if category else "_all"
    filename = f"simulation_files_{session_id[:8]}{suffix}.zip"
    return tmp_path, filename


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as exc:  # noqa: BLE001
        logger.warning("failed to remove temp bundle %s: %s", path, exc)


def _combined_bundle_response(session, session_id: str, category: str | None):
    try:
        tmp_path, filename = _build_combined_bundle(session, session_id, category)
    except _BundleBuildError as exc:
        return jsonify({"error": "bundle_unavailable", "message": exc.message}), exc.status_code

    def generate():
        try:
            with open(tmp_path, "rb") as handle:
                while True:
                    chunk = handle.read(_CHUNK)
                    if not chunk:
                        break
                    yield chunk
        finally:
            _remove_quietly(tmp_path)

    resp = Response(stream_with_context(generate()))
    resp.headers["Content-Type"] = "application/zip"
    try:
        resp.headers["Content-Length"] = str(os.path.getsize(tmp_path))
    except OSError:
        pass
    resp.headers["Content-Disposition"] = _attachment_header(filename)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp
