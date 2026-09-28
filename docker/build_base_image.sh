#!/bin/bash
# ============================================================================
# 构建 TrainMeshAgent 基础镜像（内含 Python 3.10 运行时 + 全部三方依赖）
#
# 只在依赖变化时重建，日常打包部署走 build.sh，不碰基础镜像。
# 用法：
#   bash docker/build_base_image.sh
#   BASE_TAG=repository/python3.10-aicm-base:1.1 bash docker/build_base_image.sh
#   SKIP_HCCL=1 bash docker/build_base_image.sh     # 尚未拿到 hccl whl 时
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
