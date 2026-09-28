#!/bin/sh
# ============================================================================
# 构建期 shim —— 只在 docker/base/Dockerfile 的那次 apt 安装期间顶替
# /usr/bin/{ucf,ucfr}，装完立即摘除，不进最终镜像。
#
# 为什么需要它
#   Docker < 20.10 的节点上，容器内 test -w 对任何文件都返回假（默认 seccomp
#   白名单缺 faccessat2，而 jammy 的 glibc 用 faccessat2 实现 access()，且只在
#   ENOSYS 时回退）。而 ucf / ucfr 的前置检查是
#       [ -e $statedir/{hashfile,registry} -a ! -w ... ]   → exit 1
#   于是 postgresql-common 的 postinst 必然失败、postgresql-14 连带未配置、
#   apt 返回 100（Ubuntu #1943049 / Debian #1005906）。
#
# 为什么必须"每次调用前清状态"，而不是"安装前删一次"
#   首次安装时 postgresql-common 的 postinst 会调用 ucf/ucfr 各两次：
#       setup_createclusterconf           → ucf <tmp> <ccconfig> + ucfr <pkg> <ccconfig>
#       末尾（"$2" 为空 ⇒ `dpkg --compare-versions "" lt 183~` 成立）
#                                        → ucf --purge <lrconfig> + ucfr --purge ...
#   第一次 ucf 调用就会把 hashfile 建回来，第二次调用便撞上检查；
#   而 --purge 路径里 ucfr 又要求 registry 存在（否则 `exit 6`），与「检查要求
#   文件不存在」互斥 —— 任何静态的文件布局都无法同时满足，只能逐次清理。
#
# 行为
#   1) --purge / -p：直接成功。构建期没有任何旧的 ucf 登记需要清理，与真身在
#      「无登记」时的行为（ucfr 打印 already purged 后 exit 0）等价；
#   2) 其余调用：先删掉状态文件再委托 *.real 真身 —— 检查随即跳过，
#      真身会自行重建（ucf 的 replace_md5sum、ucfr 的 touch 分支都处理了缺失）。
#
# 退出条件
#   节点 Docker 升到 >= 20.10（默认 profile 已允许 faccessat2）后，
#   本文件与 Dockerfile 里的拦截层、以及改成 test -f 的断言都可以一并回退。
# ============================================================================
for a in "$@"; do
    case "$a" in
        --purge | -p) exit 0 ;;
    esac
done

rm -f /var/lib/ucf/hashfile /var/lib/ucf/registry

case "${0##*/}" in
    ucfr) exec /usr/bin/ucfr.real "$@" ;;
    *)    exec /usr/bin/ucf.real "$@" ;;
esac
