#!/bin/bash
# ============================================================================
# 构建 TrainMeshAgent 基础镜像（内含 Python 3.10 运行时 + 全部三方依赖）
#
# 只在依赖变化时重建，日常打包部署走 build.sh，不碰基础镜像。
# 用法：
#   bash docker/build_base_image.sh
#   BASE_TAG=repository/python3.10-aicm-base:1.1 bash docker/build_base_image.sh
#   SKIP_HCCL=1 bash docker/build_base_image.sh          # 尚未拿到 hccl whl 时
#   SECCOMP=none bash docker/build_base_image.sh         # 跳过运行时探测（已打 Dockerfile 兜底补丁时用）
#   SECCOMP=unconfined bash docker/build_base_image.sh   # 强制给构建容器加 seccomp=unconfined
#
# ── 背景：Docker < 20.10 的节点上，本镜像会失败在 Step 14 ────────────────────
# 现象：
#     Setting up postgresql-common (238) ...
#     ucf: do not have write privilege to the state data
#     dpkg: error processing package postgresql-common (--configure): ... exit status 1
#     dpkg: dependency problems prevent configuration of postgresql-14 ...
#     E: Sub-process /usr/bin/dpkg returned an error code (1)   → docker build 退出 100
# 原因：老 Docker 的默认 seccomp 白名单缺少 faccessat2，而 Ubuntu 22.04 的
#   glibc(2.35) 用 faccessat2 实现 access()，且只在 ENOSYS 时回退老 syscall；
#   于是容器内 test -r/-w/-x（dash 内建）对**任何**文件都返回假。ucf 的前置检查
#   "[ -e $statedir/hashfile -a ! -w $statedir/hashfile ]" 因此恒成立并 exit 1，
#   postgresql-common 的 postinst（setup_createclusterconf 调 ucf）随之失败，
#   连带 postgresql-14 未配置 —— 与文件权限、基镜像内容都无关。
#   参见 Ubuntu #1943049、Debian #1005906（同一根因，Docker 20.10 起修好）。
#
# ── 三条可行路线（脚本不会替你做决定）────────────────────────────────────────
#   1) 在 Docker >= 20.10 的节点上构建本镜像，再 push / pull 内网 registry；
#   2) 升级本节点 Docker（根治，同时修掉 statx 一族问题）；
#   3) 给 docker/base/Dockerfile 打 ucf 兜底补丁（见 Dockerfile 中的注释），
#      之后用 SECCOMP=none 让本脚本静默。
# 注意：18.09 的 daemon 会直接拒绝构建期的 security-opt——
#   "Error response from daemon: The daemon on this platform does not support
#    setting security options on build"，所以默认不会加该 flag。
# ============================================================================
set -ex

BASE_TAG="${BASE_TAG:-repository/python3.10-aicm-base:1.0}"
SKIP_HCCL="${SKIP_HCCL:-0}"
SECCOMP="${SECCOMP:-auto}"

if ! ls docker/wheels/*.whl > /dev/null 2>&1 && [ "${SKIP_HCCL}" != "1" ]; then
    echo "提示：docker/wheels/ 下未找到 hccl_format-*.whl"
    echo "      请先放入该 whl，或设置 SKIP_HCCL=1 先构建（仿真任务将不可用）"
fi

# 从 Dockerfile 解析 FROM 镜像，用于探测容器内 access()/test -w 是否可用
BASE_IMAGE="$(sed -n 's/^[[:space:]]*FROM[[:space:]]\+\([^[:space:]]*\).*/\1/p' \
    docker/base/Dockerfile | head -n 1 | tr -d '\r')"

SECCOMP_OPT=()
case "${SECCOMP}" in
    none | "")
        echo "[preflight] SECCOMP=${SECCOMP}：跳过运行时探测，不加 --security-opt"
        ;;
    unconfined)
        echo "[preflight] SECCOMP=unconfined：构建时带上 --security-opt seccomp=unconfined"
        echo "[preflight]   注意：仅 Docker >= 20.10 的 daemon 支持该选项，18.09 会直接拒绝构建"
        SECCOMP_OPT=(--security-opt seccomp=unconfined)
        ;;
    auto)
        if [ -z "${BASE_IMAGE}" ]; then
            echo "[preflight] 未能从 docker/base/Dockerfile 解析出 FROM，跳过运行时探测" >&2
        elif docker run --rm "${BASE_IMAGE}" \
                sh -c 'echo probe > /tmp/.seccomp-probe && [ -w /tmp/.seccomp-probe ]' > /dev/null 2>&1; then
            echo "[preflight] 运行时正常：容器内 test -w 可用，无需任何兜底"
        else
            echo "======================================================================" >&2
            echo "[preflight] 运行时缺陷：容器内 test -w 对可写文件也返回假" >&2
            echo "[preflight]   docker（$(docker version -f '{{.Server.Version}}' 2>/dev/null || echo 未知)）< 20.10 的默认 seccomp 白名单缺 faccessat2，" >&2
            echo "[preflight]   于是 ucf 的前置检查恒成立并 exit 1 ⇒ postgresql-common 配置失败 ⇒ apt 100" >&2
            echo "[preflight]   本 daemon 不支持在构建期设置 security-opt，脚本不会强行加 flag。可选：" >&2
            echo "[preflight]     1) 在 Docker >= 20.10 的节点上构建，再 push / pull 内网 registry" >&2
            echo "[preflight]     2) 升级本节点 Docker（根治，同时修掉 statx 一族问题）" >&2
            echo "[preflight]     3) 给 docker/base/Dockerfile 打 ucf 兜底补丁，之后用 SECCOMP=none 静默" >&2
            echo "[preflight]   否则本次构建大概率终止在 Step 14 的 ucf 报错处；详见本文件头部注释" >&2
            echo "======================================================================" >&2
        fi
        ;;
    *)
        echo "用法错误：SECCOMP 只支持 auto / unconfined / none（当前：${SECCOMP}）" >&2
        exit 2
        ;;
esac

docker build \
    -f docker/base/Dockerfile \
    --build-arg SKIP_HCCL="${SKIP_HCCL}" \
    "${SECCOMP_OPT[@]}" \
    -t "${BASE_TAG}" \
    .

echo "基础镜像构建完成：${BASE_TAG}"
