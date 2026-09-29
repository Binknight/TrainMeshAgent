#!/bin/bash
# ============================================================================
# equivalent-modeling-service 业务镜像打包脚本
#
# 产物：deployPackages.<datetime>.tar.gz
#   ├── Dockerfile                 业务镜像 Dockerfile（FROM python3.10-aicm-base:1.0）
#   ├── equivalent-modeling-service.tgz         app / static / mcp_server / aicm / docker
#   └── chart.tgz                  helm chart（若仓库存在 charts/）
#
# 依赖已预置在基础镜像中，业务镜像构建期不联网、不装包。
# 环境变量（沿用 CI 约定）：buildNumber / isRelease / WORKSPACE
# ============================================================================
set -ex

if [ -z ${buildNumber} ];then
   if [ -e /proc/sys/kernel/random/uuid ] && [ -r /proc/sys/kernel/random/uuid ];then
       build=${RANDOM}
   else
       build=${RANDOM}
   fi
   datetime=`date +%Y%m%d%H%M%S`
   buildNumber="${datetime}.${build}"
else
   buildNumber="${buildNumber}"
fi

# 固定版本
SERVICE_VERSION="1.0.0"
# 判断当前构建是否为版本构建，以及定义构建变量(包版本,包服务名称,包编译存放路径,包类型,包编译名称,包打包名称)
if [ "${isRelease}"x = "false"x ];then
	#版本号+时间戳+build随机数写入buildInfo.properties
	echo "buildVersion=${SERVICE_VERSION}.$buildNumber">${WORKSPACE}/buildInfo.properties
elif [ "${isRelease}"x = "true"x ];then
	echo "buildVersion=${SERVICE_VERSION}">${WORKSPACE}/buildInfo.properties
fi

### 打包python应用
### 仿真工具 aicm/ 存在则一并打入（hccl whl 不重复携带，已在基础镜像中装好）
PKG_FILES="app static mcp_server docker requirements.txt"
if [ -d aicm ]; then
    PKG_FILES="${PKG_FILES} aicm"
else
    echo "警告：未找到 aicm/ 仿真工具源码，本次包内不含仿真工具（仿真任务会报 sim_tool_home not found）"
fi

### docker/ 目录整体打入，但排除只用于构建的两处内容：
###   docker/wheels/*  已装进基础镜像，无需重复携带
###   docker/base/*    基础镜像 Dockerfile，业务镜像用不到
### 用 <dir>/* 而非 <dir>，避免依赖 tar 对「目录本身匹配即跳过整棵子树」的语义差异
tar zcvf equivalent-modeling-service.tgz \
    --exclude='docker/wheels/*' \
    --exclude='docker/base/*' \
    --exclude='*/__pycache__' \
    --exclude='*/__pycache__/*' \
    --exclude='*.pyc' \
    ${PKG_FILES}

### 打包helm
if [ -d charts ]; then
    tar zcvf chart.tgz charts/*
fi

mkdir deployPackages.${datetime}
cp Dockerfile equivalent-modeling-service*.tgz deployPackages.${datetime}/
if [ -f chart.tgz ]; then
    cp chart.tgz deployPackages.${datetime}/
fi
tar zcvf deployPackages.${datetime}.tar.gz deployPackages.${datetime}

echo "打包完成：deployPackages.${datetime}.tar.gz"
