#!/bin/bash
# ============================================================================
# 构建 TrainMeshAgent 基础镜像（内含 Python 3.10 运行时 + 全部三方依赖）
#
# 只在依赖变化时重建，日常打包部署走 build.sh，不碰基础镜像。
# 用法：
#   bash docker/build_base_image.sh
#   BASE_TAG=repository/python3.10-aicm-base:1.1 bash docker/build_base_image.sh
#   SKIP_HCCL=1 bash docker/build_base_image.sh          # 尚未拿到 hccl whl 时
#
# ── 本次改造：ucf/seccomp 兜底逻辑已整体移除 ─────────────────────────────────
# 历史问题（Docker < 20.10 节点上构建必然失败）：
#     Setting up postgresql-common (238) ...
#     ucf: do not have write privilege to the state data
#     dpkg: error processing package postgresql-common (--configure): ... exit status 1
#     E: Sub-process /usr/bin/dpkg returned an error code (1)   → docker build 退出 100
# 根因：老 Docker 的默认 seccomp 白名单缺 faccessat2，而 Ubuntu 22.04 的
#   glibc(2.35) 用 faccessat2 实现 access()，于是容器内 `test -w` 对任何文件都
#   返回假；ucf 的前置检查因此恒成立并 exit 1，postgresql-common 的 postinst
#   随之失败，连带 postgresql-14 未配置。
#
# **该问题已随 `postgresql-14` 的移除而消失**：数据库改为 SQLite 后不再安装
# postgresql-common/ucf，因此不再需要 ucf shim（docker/base/ucf-shim.sh 已删除）
# 与 `--security-opt seccomp=unconfined` 这一族兜底。脚本相应退回最简形态。
#
# 若将来因别的 apt 包再次撞上同一 seccomp 缺陷（现象同上是 `test -w` 恒假），
# 根治办法仍是升级节点 Docker 到 >= 20.10，或在旧节点上用 20.10+ 构建后推镜像。
# ============================================================================
set -ex

BASE_TAG="${BASE_TAG:-repository/python3.10-aicm-base:1.0}"
SKIP_HCCL="${SKIP_HCCL:-0}"

if ! ls docker/wheels/*.whl > /dev/null 2>&1 && [ "${SKIP_HCCL}" != "1" ]; then
    echo "提示：docker/wheels/ 下未找到 hccl_format-*.whl"
    echo "      请先放入该 whl，或设置 SKIP_HCCL=1 先构建（仿真任务将不可用）"
fi

docker build \
    -f docker/base/Dockerfile \
    --build-arg SKIP_HCCL="${SKIP_HCCL}" \
    -t "${BASE_TAG}" \
    .

echo "基础镜像构建完成：${BASE_TAG}"
