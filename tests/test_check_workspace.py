"""仿真工作区启动自检（docker/check_workspace.py）回归测试。

覆盖：
  - /proc/self/mountinfo 解析（含路径的八进制转义）
  - 挂载点判定、目录不存在 / 不可写的报错
  - AICM_MCP_WORKSPACE_ALLOW_LOCAL=1 的豁免行为
  - main() 退出码：未挂载时必须退非 0，否则就会静默把产物写进镜像层

Run: python tests/test_check_workspace.py
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 工作目录固定在仓库 .tmp/ 下（已 gitignore）：某些沙箱不允许写系统临时目录
TMP_ROOT = REPO / ".tmp" / "check_workspace_test"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_failures: list[str] = []
_case_index = 0


@contextlib.contextmanager
def tmp_dir():
    """在本用例专属目录下开一个临时目录。

    不能用 tempfile.TemporaryDirectory：它以 0o700 建目录，在部分环境（含本仓库
    开发机的文件沙箱）该目录会变成不可写，属于环境差异、与被测逻辑无关。
    同理，工作目录也不能用系统临时目录（可能不可写）。
    """
    global _case_index
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    _case_index += 1
    path = TMP_ROOT / f"case{_case_index}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "check_workspace", REPO / "docker" / "check_workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_escape(path: str) -> str:
    """按 /proc/self/mountinfo 的内核规则转义路径（空格 → \\040 等）。"""
    for plain, escaped in (("\\", "\\134"), (" ", "\\040"), ("\t", "\\011"), ("\n", "\\012")):
        path = path.replace(plain, escaped)
    return path


def write_mountinfo(path: Path, mount_points) -> Path:
    """写一个最小可用的 mountinfo（字段顺序同 /proc/self/mountinfo）。"""
    lines = ["21 26 0:20 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw"]
    for index, point in enumerate(mount_points):
        lines.append(f"{100 + index} 26 8:1 / {kernel_escape(str(point))} rw,relatime - ext4 /dev/sda1 rw")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@contextlib.contextmanager
def env(**values):
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_main(module, **environment):
    """跑一次 main()，返回 (退出码, stdout)。"""
    buffer = io.StringIO()
    with env(**environment), contextlib.redirect_stdout(buffer):
        code = module.main()
    return code, buffer.getvalue()


def main() -> None:
    module = load_module()

    print("== mountinfo 解析 ==")
    with tmp_dir() as tmp:
        tmp_path = Path(tmp)
        mountinfo = write_mountinfo(
            tmp_path / "mountinfo",
            ["/mnt2", "/data/aicm/with space"],
        )
        points = module.read_mount_points(str(mountinfo))
        expected = {"/proc", "/mnt2", "/data/aicm/with space"}
        check("解析出全部挂载点", points == expected, repr(points))
        check("路径中的空格按内核规则转义后可还原", "/data/aicm/with space" in points, repr(points))
        check("mountinfo 缺失时返回空集", module.read_mount_points(str(tmp_path / "nope")) == set())

    with tmp_dir() as tmp:
        workspace = Path(tmp) / "ws"
        workspace.mkdir()
        listed = write_mountinfo(Path(tmp) / "listed", [str(workspace)])
        unlisted = write_mountinfo(Path(tmp) / "unlisted", ["/mnt2"])

        print("== 挂载点判定 ==")
        check("已挂载路径判定为 True", module.is_mount_point(workspace, str(listed)) is True)
        check("未挂载路径判定为 False", module.is_mount_point(workspace, str(unlisted)) is False)
        check(
            "mountinfo 不可读时判定为 False（不能证明已挂载）",
            module.is_mount_point(workspace, str(Path(tmp) / "missing")) is False,
        )

        print("== 可写性 ==")
        check("可写目录无报错", module.writability_error(workspace) == "")
        check(
            "自检不留下临时文件",
            [p.name for p in workspace.iterdir()] == [],
            repr([p.name for p in workspace.iterdir()]),
        )

        print("== check() ==")
        check("目录不存在 → 致命", module.check(workspace / "nope", False, str(listed)) != [])
        not_mounted = module.check(workspace, False, str(unlisted))
        check("未挂载 + 不允许本地 → 致命", len(not_mounted) == 1 and "不是挂载点" in not_mounted[0], repr(not_mounted))
        check("未挂载 + 允许本地 → 通过", module.check(workspace, True, str(unlisted)) == [])
        check("已挂载 → 通过", module.check(workspace, False, str(listed)) == [])

        print("== main() 退出码 ==")
        original = module.MOUNTINFO_PATH
        try:
            module.MOUNTINFO_PATH = str(listed)
            code, out = run_main(
                module,
                AICM_MCP_WORKSPACE_ROOT=str(workspace),
                AICM_MCP_WORKSPACE_ALLOW_LOCAL=None,
            )
            check("已挂载 → 退出码 0", code == 0, f"code={code} out={out!r}")
            check("启动日志含挂载点确认", "是挂载点" in out, repr(out))
            check("启动日志含剩余空间", "剩余空间" in out, repr(out))

            module.MOUNTINFO_PATH = str(unlisted)
            code, out = run_main(
                module,
                AICM_MCP_WORKSPACE_ROOT=str(workspace),
                AICM_MCP_WORKSPACE_ALLOW_LOCAL=None,
            )
            check("未挂载 → 退出码 1（关键兜底）", code == 1, f"code={code} out={out!r}")
            check("报错说明产物会落在镜像层", "镜像层" in out, repr(out))

            code, out = run_main(
                module,
                AICM_MCP_WORKSPACE_ROOT=str(workspace),
                AICM_MCP_WORKSPACE_ALLOW_LOCAL="1",
            )
            check("未挂载 + 豁免 → 退出码 0", code == 0, f"code={code} out={out!r}")
            check("豁免时有醒目告警", "警告" in out, repr(out))

            code, out = run_main(
                module,
                AICM_MCP_WORKSPACE_ROOT=str(Path(tmp) / "does-not-exist"),
                AICM_MCP_WORKSPACE_ALLOW_LOCAL="1",
            )
            check("目录不存在时豁免也无效 → 退出码 1", code == 1, f"code={code} out={out!r}")
        finally:
            module.MOUNTINFO_PATH = original

    print("== 真实 mountinfo（仅 Linux） ==")
    real_mountinfo = Path("/proc/self/mountinfo")
    if real_mountinfo.is_file():
        check("根目录 / 是挂载点", module.is_mount_point(Path("/"), str(real_mountinfo)) is True)
        check(
            "不存在的路径不是挂载点",
            module.is_mount_point(Path("/nonexistent-aicm-probe"), str(real_mountinfo)) is False,
        )
    else:
        print("  [SKIP] 当前平台无 /proc/self/mountinfo")

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        sys.exit(1)
    print("  [PASS] 工作区自检行为符合预期（未挂载即失败，不静默写镜像层）。")


if __name__ == "__main__":
    main()
