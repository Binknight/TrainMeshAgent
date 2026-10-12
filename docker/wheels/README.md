# docker/wheels/ —— 仿真工具随目录下发的离线 wheel

把 **`hccl_format-0.1-py3-none-any.whl`** 放到本目录，基础镜像构建时会自动安装：

```
docker/wheels/hccl_format-0.1-py3-none-any.whl
```

基础镜像 `docker/base/Dockerfile` 用 `COPY docker/wheels/ /tmp/wheels/` 取包，并以
`pip3 install /tmp/wheels/*.whl` 安装（不经索引，纯本地）。

## 注意

- 若本目录**为空**且未加 `--build-arg SKIP_HCCL=1`，基础镜像构建会**主动失败**并提示，
  避免缺少仿真依赖的镜像流到业务层。
- 该 whl 是 `py3-none-any`（纯 Python、无 ABI、无架构标签），因此在 Python 3.10 /
  x86_64 上必然可装，无需编译。
- 本目录不进业务镜像：`build.sh` 打包时会 `--exclude=docker/wheels`，
  因为包已在基础镜像中装好，无需重复携带。
