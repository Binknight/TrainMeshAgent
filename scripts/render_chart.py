"""Helm chart 近似渲染校验（本机未安装 helm）。

为什么不用 Jinja2：
    Helm 用 Go text/template，`.Values.a.b` 的点号前缀与 `|` 管道语义跟 Jinja2
    的过滤器语法在语言层面冲突（Jinja2 一律报 "unexpected '.'"）。所以这里实现
    一个够用的极简渲染器，只覆盖本 chart 实际用到的语法子集：
        {{ .Values.a.b.c }} / {{ .Values.a.b | quote }} / {{ .Values.a.b | int }}
        {{ .Values.x | default 1 }} / {{ if .Values.x }}...{{ end }}
        {{ if eq .Values.x "s"}} ... {{ else if eq ... }} ... {{ end }}
        {{- ... }} 与 {{ ... }} 的空行裁剪
    目的不是替代 helm，而是抓住本次改动的两类高危错误：
        1. 模板引用了 values.yaml 里不存在的键（删了 secrets.databaseUrl、
           新增 database.* / config.pgSocketDir，正是高危区）
        2. 渲染产物不是合法 YAML / 占位符没被替换干净 / 关键字段缺失

覆盖两种部署形态：
    A. 默认（内嵌数据库）—— 不应出现 DATABASE_URL
    B. 外部数据库回退 —— 必须出现 DATABASE_URL（逃生门可用）
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHART = ROOT / "charts"
TEMPLATES = sorted((CHART / "templates").glob("*.yaml"))

ACTION_RE = re.compile(r"\{\{-?\s*(.*?)\s*-?\}\}")


class RenderError(Exception):
    pass


def get_path(scope: dict, expr: str):
    """解析 .Values.a.b / a.b 形式的取值；缺失返回 None。"""
    expr = expr.strip()
    if expr.startswith("."):
        expr = expr[1:]
    cur = scope
    for part in expr.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def apply_pipe(value, pipe: str, origin: str = ""):
    """处理 | quote / | int / | default X（可串联）。"""
    for stage in pipe.split("|"):
        stage = stage.strip()
        if not stage:
            continue
        if stage == "quote":
            value = '"%s"' % value
        elif stage == "int":
            if value is None:
                raise RenderError(f"{origin} 取值为空，无法 int()（键名可能写错）")
            value = str(int(value))
        elif stage.startswith("default"):
            fallback = stage.split(None, 1)[1].strip() if len(stage.split(None, 1)) > 1 else ""
            if value in (None, "", False):
                value = fallback
        else:
            raise RenderError(f"{origin} 不支持的管道: {stage}")
    return value


def eval_expr(expr: str, scope: dict):
    """求值单个动作表达式（不含 if/else/end）。"""
    expr = expr.strip()
    if expr.startswith("if ") or expr in ("else", "end"):
        raise RenderError(f"eval_expr 不应收到控制流: {expr}")
    try:
        if "|" in expr:
            head, _, pipe = expr.partition("|")
            return apply_pipe(eval_expr(head, scope), pipe, origin=expr)
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        if re.fullmatch(r"-?\d+", expr):
            return int(expr)
        return get_path(scope, expr)
    except RenderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RenderError(f"表达式 {expr!r} 求值失败: {type(exc).__name__}: {exc}") from exc


def condition(expr: str, scope: dict) -> bool:
    """处理 if 条件：eq a b / 裸取值真值判定。"""
    expr = expr.strip()
    m = re.fullmatch(r"eq\s+(.+?)\s+(\"[^\"]*\"|\S+)", expr)
    if m:
        left = eval_expr(m.group(1), scope)
        right = eval_expr(m.group(2), scope)
        return str(left) == str(right)
    value = eval_expr(expr, scope)
    return bool(value)


def render(template: str, scope: dict) -> str:
    """逐行渲染，用栈跟踪 if / else if / else / end 的活跃状态。

    栈帧字段：
        parent_active  外层是否活跃（外层被跳过时，整条 if 链都不生效）
        taken          本 if 链是否已有分支命中过
        cond_active    当前分支自身的条件是否成立
    行是否输出 = parent_active and cond_active。
    """
    out: list[str] = []
    stack: list[dict] = []
    pos = 0
    for match in ACTION_RE.finditer(template):
        body = match.group(1).strip()

        is_cond = body.startswith("if ") or body.startswith("else if ")
        if is_cond:
            # 把动作之前的静态文本写出去（仅当当前处于活跃分支）
            out.append(_emit(template[pos:match.start()], stack))
            cond = condition(body[3:] if body.startswith("if ") else body[len("else if "):], scope)
            if body.startswith("if "):
                parent_active = _is_active(stack)
                stack.append({"parent_active": parent_active, "taken": cond, "cond_active": cond})
            else:
                if not stack:
                    raise RenderError("else if 无匹配的 if")
                frame = stack[-1]
                frame["cond_active"] = (not frame["taken"]) and cond
                frame["taken"] = frame["taken"] or cond
        elif body == "else":
            out.append(_emit(template[pos:match.start()], stack))
            if not stack:
                raise RenderError("else 无匹配的 if")
            frame = stack[-1]
            frame["cond_active"] = not frame["taken"]
            frame["taken"] = True
        elif body == "end":
            out.append(_emit(template[pos:match.start()], stack))
            if not stack:
                raise RenderError("end 无匹配的 if")
            stack.pop()
        else:
            # 普通取值：仅在活跃分支里求值
            out.append(_emit(template[pos:match.start()], stack))
            if _is_active(stack):
                out.append(str(eval_expr(body, scope)))
        pos = match.end()

    if stack:
        raise RenderError(f"有 {len(stack)} 个 if 未闭合")
    out.append(_emit(template[pos:], stack))
    return "".join(out)


def _is_active(stack: list[dict]) -> bool:
    return all(frame["parent_active"] and frame["cond_active"] for frame in stack)


def _emit(text: str, stack: list[dict]) -> str:
    """活跃时才输出这段静态文本；被跳过的分支整段丢弃（含其换行）。"""
    return text if _is_active(stack) else ""


def load_values(raw_text: str) -> dict:
    """替换平台占位符（@repository@ 等）再解析 —— 裸 @ 不是合法 YAML 起始字符。"""
    substituted = re.sub(
        r"@([A-Za-z0-9_.]+)@",
        lambda m: '"PLACEHOLDER_' + m.group(1).replace(".", "_").upper() + '"',
        raw_text,
    )
    return yaml.safe_load(substituted)


def check_referenced_keys(values: dict) -> list[str]:
    """静态提取模板引用的 .Values.a.b.c，逐个查存在性（带 if 保护的可选键除外）。"""
    errors = []
    pattern = re.compile(r"\.Values\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
    optional = {"secrets.databaseUrl", "secrets.externalProxy"}
    scope = {"Values": values}
    for path in TEMPLATES:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for ref in pattern.findall(line):
                if ref in optional:
                    continue
                if get_path(scope, f"Values.{ref}") is None:
                    errors.append(
                        f"{path.name}:{lineno} 引用了 values 中不存在/为空的键 .Values.{ref}"
                    )
    return errors


def render_all(values: dict, label: str):
    rendered, errors = {}, []
    scope = {"Values": values}  # 模板以 .Values.xxx 取值，作用域需包一层
    for path in TEMPLATES:
        try:
            out = render(path.read_text(encoding="utf-8"), scope)
        except RenderError as exc:
            errors.append(f"[{label}] {path.name} 渲染失败: {exc}")
            continue
        rendered[path.name] = out
        try:
            docs = [d for d in yaml.safe_load_all(out)]
        except yaml.YAMLError as exc:
            errors.append(f"[{label}] {path.name} 渲染结果不是合法 YAML: {exc}")
            continue
        if not any(docs):
            errors.append(f"[{label}] {path.name} 渲染为空文档")
        leftovers = re.findall(r"\{\{.*?\}\}", out)
        if leftovers:
            errors.append(f"[{label}] {path.name} 有未解析片段: {leftovers[:3]}")
    return rendered, errors


def main() -> int:
    values = load_values((CHART / "values.yaml").read_text(encoding="utf-8"))
    print(f"[0] values.yaml 顶层键: {sorted(values)}")
    print(f"[0] 模板: {[p.name for p in TEMPLATES]}")

    errors = check_referenced_keys(values)
    print(f"[1] 键引用检查: {'OK' if not errors else str(len(errors)) + ' 个问题'}")

    print("\n[2] 形态 A —— 默认（内嵌数据库）")
    rendered_a, errs_a = render_all(values, "A")
    errors += errs_a
    for name, out in rendered_a.items():
        print(f"    {name}: {len(out.splitlines())} 行，渲染 + YAML 解析通过")

    print("\n[3] 形态 B —— 外部 PostgreSQL 回退")
    values_b = {k: (dict(v) if isinstance(v, dict) else v) for k, v in values.items()}
    values_b["secrets"] = dict(values_b.get("secrets", {}))
    values_b["secrets"]["databaseUrl"] = "postgresql://user:pass@10.1.2.3:5432/train_mesh_agent"
    rendered_b, errs_b = render_all(values_b, "B")
    errors += errs_b

    def has_active_key(doc: str, key: str) -> bool:
        """只看未被注释的 YAML 键行，避免注释里提到 DATABASE_URL 造成误判。"""
        return any(
            re.match(rf"^\s*{key}\s*:", line) for line in doc.splitlines()
        )

    if "secret.yaml" in rendered_a:
        if has_active_key(rendered_a["secret.yaml"], "DATABASE_URL"):
            errors.append("[A] 默认形态不应渲染出 DATABASE_URL（应回退镜像内默认值）")
        else:
            print("    OK: 形态 A 无 DATABASE_URL 键 -> 回退镜像内嵌默认值")
    if "secret.yaml" in rendered_b:
        if not has_active_key(rendered_b["secret.yaml"], "DATABASE_URL"):
            errors.append("[B] 外部 PG 形态必须渲染出 DATABASE_URL")
        else:
            print("    OK: 形态 B 渲染出 DATABASE_URL 键 -> 外部 PG 逃生门可用")

    dep = rendered_a.get("deployment.yaml", "")
    if dep:
        checks = {
            "sim-db 卷定义": "name: sim-db" in dep,
            "hostPath 节点路径 /data/aicm/db": "path: /data/aicm/db" in dep,
            "DirectoryOrCreate ≥2（workspace + db）": dep.count("DirectoryOrCreate") >= 2,
            "容器内挂载点 /home/aicm/db 出现 ≥3 次（init chown + init mount + 主容器 mount）":
                dep.count("/home/aicm/db") >= 3,
        }
        for label, ok in checks.items():
            print(f"    {'OK  ' if ok else 'FAIL'}: deployment {label}")
            if not ok:
                errors.append(f"[A] deployment.yaml 缺少 {label}")

    cm = rendered_a.get("configMap.yaml", "")
    if cm:
        ok = "PG_SOCKET_DIR: " in cm and "/home/aicm/db/run" in cm
        print(f"    {'OK  ' if ok else 'FAIL'}: configMap 下发 PG_SOCKET_DIR")
        if not ok:
            errors.append("[A] configMap.yaml 缺少 PG_SOCKET_DIR 或取值不对")

    if errors:
        print(f"\n发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: chart 渲染、键引用与关键字段检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
