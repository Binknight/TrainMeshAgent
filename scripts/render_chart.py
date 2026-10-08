"""Helm chart 近似渲染校验（本机未安装 helm）。

为什么不用 Jinja2：
    Helm 用 Go text/template，`.Values.a.b` 的点号前缀与 `|` 管道语义跟 Jinja2
    的过滤器语法在语言层面冲突（Jinja2 一律报 "unexpected '.'"）。所以这里实现
    一个够用的极简渲染器，只覆盖本 chart 实际用到的语法子集：
        {{ .Values.a.b.c }} / {{ .Values.a.b | quote }} / {{ .Values.a.b | int }}
        {{ .Values.x | default 1 }} / {{ if .Values.x }}...{{ end }}
        {{ if eq .Values.x "s"}} ... {{ else if eq ... }} ... {{ end }}
        {{- ... }} 与 {{ ... }} 的空行裁剪
    目的不是替代 helm，而是抓住本次改动的三类高危错误：
        1. 模板引用了 values.yaml 里不存在的键（删了 config.pgSocketDir、
           新增 config.sqlitePath / database.*，正是高危区）
        2. 渲染产物不是合法 YAML / 占位符没被替换干净 / 关键字段缺失
        3. **ConfigMap/Secret 里出现 null 值**（渲染成 `KEY:`）—— k8s 会以
           `unknown object type "nil"` 拒绝发布，且本机没有 helm 时最难自查。
           历史事故：平台未登记 `openaiApiKey` 占位符 → 空串 → nil → 发布失败。

覆盖三种部署形态：
    A. 默认（平台变量已登记）—— ConfigMap 不应出现 DATABASE_URL 键，且必须出现
       非空 OPENAI_API_KEY（本 chart 已删除 secret.yaml，LLM 四个键经 ConfigMap 下发）
    B. 外部 PostgreSQL 逃生门（config.databaseUrl）—— ConfigMap 必须出现 DATABASE_URL 键
    C. 平台变量一个都没登记（四个 LLM 键全为空）—— 必须**不下发**任何 OPENAI_* 键，
       而不是渲染出 null（2026-09-30 事故的直接复现条件）
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHART = ROOT / "charts" / "equivalent-modeling-service"
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
            # 必须与 Sprig 对齐：quote(nil) 返回**空串**（不是 "None"，也不是
            # Go 模板的 "<no value>"），于是模板只剩 `KEY:` → YAML null。
            # 本地渲染器若把 nil 渲染成 "None" 就会漏掉这类事故 —— 历史上
            # OPENAI_API_KEY 的发布失败正是这样漏过去的。
            value = "" if value is None else '"%s"' % value
        elif stage == "int":
            if value is None:
                raise RenderError(f"{origin} 取值为空，无法 int()（键名可能写错）")
            value = str(int(value))
        elif stage in ("int64", "float64"):
            # deployment.yaml 用 `| int64` 把数值形态的镜像 tag 转回整数：
            # Helm 加载 values 时纯数字会变成 float64，Go 模板按 %v 渲染大
            # 浮点数就是科学计数法（20261008092605 -> 2.0261008092605e+13）。
            # 这里必须在本地复现同样的语义，否则该回归在 CI 里静默通过。
            if value is None:
                raise RenderError(f"{origin} 取值为空，无法 {stage}()（键名可能写错）")
            value = float(value) if stage == "float64" else int(float(value))
        elif stage.startswith("default"):
            fallback = stage.split(None, 1)[1].strip() if len(stage.split(None, 1)) > 1 else ""
            if value in (None, "", False):
                value = fallback
        else:
            raise RenderError(f"{origin} 不支持的管道: {stage}")
    return value


def go_kind(value) -> str:
    """把 Python 值映射成 Go 反射的 Kind 名（模板里 kindIs 的语义）。

    注意 bool 必须在 int 之前判断：Python 的 bool 是 int 的子类，
    否则 true/false 会被判成 "int"。
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float64"
    if isinstance(value, dict):
        return "map"
    if isinstance(value, (list, tuple)):
        return "slice"
    if value is None:
        return "invalid"
    return type(value).__name__


def eval_expr(expr: str, scope: dict):
    """求值单个动作表达式（不含 if/else/end）。"""
    expr = expr.strip()
    if expr.startswith("if ") or expr in ("else", "end"):
        raise RenderError(f"eval_expr 不应收到控制流: {expr}")
    try:
        # kindIs "string" .Values.x —— deployment.yaml 用它区分「数字形态的
        # 镜像 tag（需 int64 归一化）」与「字符串形态（原样下发）」。
        m = re.fullmatch(r'kindIs\s+"([A-Za-z0-9_]+)"\s+(.+)', expr)
        if m:
            return go_kind(eval_expr(m.group(2), scope)) == m.group(1)
        if "|" in expr:
            head, _, pipe = expr.partition("|")
            return apply_pipe(eval_expr(head, scope), pipe, origin=expr)
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        if re.fullmatch(r"-?\d+", expr):
            return int(expr)
        # 取值必须是 .Values.a.b 形态。**不要**在这里放行未知表达式：
        # templates/ 下连 YAML 注释里的模板记号也会被 Go 模板引擎执行（真 helm 会直接
        # 语法报错），放行未知表达式会让这类错误在本地静默通过 —— 实测就漏过一次。
        if not re.fullmatch(r"\.?[A-Za-z_][A-Za-z0-9_.]*", expr):
            raise RenderError(
                f"无法解析的表达式: {expr!r}（只支持 .Values.a.b / \"字符串\" / 整数；"
                f"管道只支持 quote|int|default。注意：注释里写成 {{{{ }}}} 的内容同样会被模板引擎执行）"
            )
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
    """替换平台占位符（@repository@ 等）再解析 —— 裸 @ 不是合法 YAML 起始字符。

    占位符**可能已经带引号**（LLM 四个键刻意写成 "@config.openaiModel@"，这样平台
    替换出的值一定是字符串而非 YAML 布尔/数字）。这类写法要连引号一起替换，否则会
    拼成 `""PLACEHOLDER_...""` 而解析失败。正则同时对齐平台那句
    `grep -oP '@\\w+(\\.\\w+)*@'`：首字符必须是名字字符，所以注释里的 `@...@`
    不会被当成占位符。
    """
    substituted = re.sub(
        r'"?@([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)@"?',
        lambda m: '"PLACEHOLDER_' + m.group(1).replace(".", "_").upper() + '"',
        raw_text,
    )
    return yaml.safe_load(substituted)


def check_referenced_keys(values: dict) -> list[str]:
    """静态提取模板引用的 .Values.a.b.c，逐个查存在性（带 if 保护的可选键除外）。"""
    errors = []
    pattern = re.compile(r"\.Values\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
    # 这四个 LLM 键与两个逃生门键都是**条件渲染**的：未登记/未启用时整行不出现，
    # 因此 values.yaml 里即使写成占位符（解析成本地渲染器用的 PLACEHOLDER_*）或缺失，
    # 也不该报「键不存在」。
    optional = {
        "config.databaseUrl",
        "config.externalProxy",
        "config.openaiBaseUrl",
        "config.openaiModel",
        "config.openaiSslVerify",
        "config.openaiApiKey",
    }
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


def check_no_null_values(rendered: dict, label: str) -> list:
    """ConfigMap.data / Secret.stringData 里的 null 或非字符串都是发布级故障。

    YAML 的 `KEY:`（冒号后无值）会被解析成 None，k8s 校验直接报
    `unknown object type "nil" in ConfigMap.data.KEY` —— 报错点离根因（占位符没被
    替换、values 里该键是 nil）很远，所以必须在渲染阶段拦下。ConfigMap 的 data
    还额外要求**所有值都是字符串**（YAML 数字/布尔会让 k8s 反序列化失败）。
    """
    errors = []
    for name, out in rendered.items():
        try:
            docs = list(yaml.safe_load_all(out))
        except yaml.YAMLError:
            continue  # 已在 render_all 里报过
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            for field in ("data", "stringData"):
                section = doc.get(field)
                if not isinstance(section, dict):
                    continue
                for key, val in section.items():
                    if not isinstance(val, str):
                        errors.append(
                            f"[{label}] {name} 的 {field}.{key} 不是字符串（{val!r}）"
                            f'—— k8s 会以 unknown object type "nil" 拒绝发布'
                        )
    return errors


def main() -> int:
    values = load_values((CHART / "values.yaml").read_text(encoding="utf-8"))
    print(f"[0] values.yaml 顶层键: {sorted(values)}")
    print(f"[0] 模板: {[p.name for p in TEMPLATES]}")

    errors = check_referenced_keys(values)
    print(f"[1] 键引用检查: {'OK' if not errors else str(len(errors)) + ' 个问题'}")

    # 四个 LLM 键的占位符**必须带引号**。流水线是纯文本 sed 替换：若写成
    # `openaiSslVerify: @config.openaiSslVerify@`，平台给 false 会得到 YAML 布尔
    # false → `{{- if }}` 判假 → 该键被静默丢弃（想关 SSL 结果仍走默认 true）。
    # 带引号可保证替换结果始终是字符串，也避开纯数字/日期形态被解析成其他类型。
    raw_values = (CHART / "values.yaml").read_text(encoding="utf-8")
    quoted_ok = True
    for llm_key in ("openaiBaseUrl", "openaiModel", "openaiSslVerify", "openaiApiKey"):
        if not re.search(rf'(?m)^\s*{llm_key}:\s*"@[^"]+@"\s*$', raw_values):
            quoted_ok = False
            errors.append(
                f'values.yaml 的 config.{llm_key} 占位符必须写成 "@...@"（带引号）'
                "：纯文本替换后要保持 YAML 字符串，否则 false/数字 会被解析成非字符串而丢键"
            )
    if quoted_ok:
        print("    OK: 四个 LLM 占位符均带引号（替换后仍是字符串）")

    # secret.yaml 已按 2026-09-30 发布事故的复盘结论删除：LLM 四个键改为由平台变量
    # 注入、经 ConfigMap 条件渲染下发。这里挡住「有人顺手把 Secret 加回来」——
    # 加回来就要重新面对 nil 值问题。
    if (CHART / "templates" / "secret.yaml").exists():
        errors.append(
            "templates/secret.yaml 已删除（LLM 键改用 ConfigMap 下发），不应重新出现"
        )

    print("\n[2] 形态 A —— 默认（内嵌数据库）")
    rendered_a, errs_a = render_all(values, "A")
    errors += errs_a
    errors += check_no_null_values(rendered_a, "A")
    for name, out in rendered_a.items():
        print(f"    {name}: {len(out.splitlines())} 行，渲染 + YAML 解析通过")

    print("\n[3] 形态 B —— 外部 PostgreSQL 回退")
    values_b = {k: (dict(v) if isinstance(v, dict) else v) for k, v in values.items()}
    values_b["config"] = dict(values_b.get("config", {}))
    values_b["config"]["databaseUrl"] = "postgresql://user:pass@10.1.2.3:5432/equivalent_modeling_service"
    rendered_b, errs_b = render_all(values_b, "B")
    errors += errs_b
    errors += check_no_null_values(rendered_b, "B")

    def has_active_key(doc: str, key: str) -> bool:
        """只看未被注释的 YAML 键行，避免注释里提到 DATABASE_URL 造成误判。"""
        return any(
            re.match(rf"^\s*{key}\s*:", line) for line in doc.splitlines()
        )

    if "configMap.yaml" in rendered_a:
        if has_active_key(rendered_a["configMap.yaml"], "DATABASE_URL"):
            errors.append("[A] 默认形态不应渲染出 DATABASE_URL（应回退镜像内默认值）")
        else:
            print("    OK: 形态 A 无 DATABASE_URL 键 -> 回退镜像内嵌默认值")
    if "configMap.yaml" in rendered_b:
        if not has_active_key(rendered_b["configMap.yaml"], "DATABASE_URL"):
            errors.append("[B] 外部 PG 形态必须渲染出 DATABASE_URL")
        else:
            print("    OK: 形态 B 渲染出 DATABASE_URL 键 -> 外部 PG 逃生门可用")

    # OPENAI_API_KEY 的值来自平台变量 config.openaiApiKey，由 ConfigMap 下发：
    # secret.yaml 删除后它是唯一来源。变量已登记时必须渲染出来（缺失时部署照样成功、
    # 只在首次 LLM 调用时报鉴权失败，静默得多，故在此断言）。
    if "configMap.yaml" in rendered_a:
        try:
            cm_doc = next(d for d in yaml.safe_load_all(rendered_a["configMap.yaml"]) if d)
            api_key = (cm_doc.get("data") or {}).get("OPENAI_API_KEY")
        except (yaml.YAMLError, StopIteration):
            api_key = None
        if not isinstance(api_key, str) or not api_key:
            errors.append(
                "[A] configMap.yaml 未下发非空 OPENAI_API_KEY"
                "（secret.yaml 已删除，LLM Key 只能来自 config.openaiApiKey）"
            )
        else:
            print(f"    OK: 形态 A 下发 OPENAI_API_KEY（{len(api_key)} 字符）")

    # ── 形态 C：平台变量一个都没登记（四个 LLM 键全为空）──
    # 这是 2026-09-30 事故的直接复现条件（占位符被替换成空串）。四个键都条件渲染，
    # 因此必须表现为「一个 OPENAI_* 键都不下发」，绝不能渲染出 null。
    print("\n[4] 形态 C —— 平台变量全未登记（四个 LLM 键都为空）")
    values_c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in values.items()}
    values_c["config"] = dict(values_c["config"])
    for llm_key in ("openaiBaseUrl", "openaiModel", "openaiSslVerify", "openaiApiKey"):
        values_c["config"][llm_key] = ""
    rendered_c, errs_c = render_all(values_c, "C")
    errors += errs_c
    errors += check_no_null_values(rendered_c, "C")
    if "configMap.yaml" in rendered_c:
        try:
            cm_c = next(d for d in yaml.safe_load_all(rendered_c["configMap.yaml"]) if d)
            data_c = cm_c.get("data") or {}
        except (yaml.YAMLError, StopIteration):
            data_c = {}
        leaked_llm = sorted(k for k in data_c if k.startswith("OPENAI_"))
        if leaked_llm:
            errors.append(
                f"[C] 变量未登记时不应下发 {leaked_llm}"
                "（空值必须表现为键不存在，不能渲染成 null/空值）"
            )
        else:
            print("    OK: 形态 C 不下发任何 OPENAI_* 键 -> 回退 app/config.py 默认值，且无 nil 值")

    dep = rendered_a.get("deployment.yaml", "")
    if dep:
        checks = {
            "sim-db 卷定义": "name: sim-db" in dep,
            "hostPath 节点路径 /data/aicm/db": "path: /data/aicm/db" in dep,
            "DirectoryOrCreate ≥2（workspace + db）": dep.count("DirectoryOrCreate") >= 2,
            "容器内挂载点 /home/data/db 出现 ≥3 次（init chown + init mount + 主容器 mount）":
                dep.count("/home/data/db") >= 3,
        }
        for label, ok in checks.items():
            print(f"    {'OK  ' if ok else 'FAIL'}: deployment {label}")
            if not ok:
                errors.append(f"[A] deployment.yaml 缺少 {label}")

    cm = rendered_a.get("configMap.yaml", "")
    if cm:
        # 改造后 configMap 下发的是 SQLite 数据文件路径（原先下发 PG socket 目录）。
        # 断言两件事：键存在，且路径落在 database.containerPath 挂载目录之内。
        ok = "SQLITE_PATH: " in cm and "/home/data/db/" in cm
        print(f"    {'OK  ' if ok else 'FAIL'}: configMap 下发 SQLITE_PATH（落在 db 挂载目录内）")
        if not ok:
            errors.append("[A] configMap.yaml 缺少 SQLITE_PATH 或取值不在 /home/data/db/ 下")

    # ── 形态 D：镜像 tag 的两种值形态都必须渲染正确 ──
    # 事故：平台把 tag 以**数字**形态注入 —— Helm 3 加载 values 会做 YAML->JSON
    # 往返，纯数字被解析成 float64，Go 模板按 %v 渲染大浮点数即科学计数法。
    # 现象极具误导性：私有仓里 tag 是正确的 20261008092605，部署出来却是
    # 2.0261008092605e+13（k8s 去拉不存在的 tag）。deployment.yaml 用
    # kindIs 分支兜住：数字走 int64，字符串原样。这里把两种形态都断言一遍，
    # 并额外覆盖带点版本 —— Sprig 的 int 会把它静默变成 0，故不能用无条件 int。
    print("\n[5] 镜像 tag 归一化（数字形态不得渲染成科学计数法）")
    image_re = re.compile(r"(?m)^\s*image:\s*(\S+)\s*$")
    tag_cases = (
        ("float64（Helm 加载纯数字 tag 后的形态）", float(20261008092605), "20261008092605"),
        ("int", 20261008092605, "20261008092605"),
        ("纯数字字符串（占位符带引号替换后）", "20261008092605", "20261008092605"),
        ("带点版本字符串", "1.0.0.20250920", "1.0.0.20250920"),
    )
    for label, tag, expect in tag_cases:
        values_d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in values.items()}
        values_d["images"] = dict(values_d["images"])
        values_d["images"]["version"] = tag
        rendered_d, errs_d = render_all(values_d, f"tag:{label}")
        errors += errs_d
        images = image_re.findall(rendered_d.get("deployment.yaml", ""))
        if len(images) != 2:
            errors.append(f"[5] {label}: 期望 2 处 image 字段（initContainer + 主容器），实际 {len(images)} 处")
            continue
        for img in images:
            if img.endswith(":" + expect) and "e+" not in img:
                print(f"    OK  : {label} -> {img}")
            else:
                errors.append(
                    f"[5] {label}: 镜像渲染为 {img}（期望以 :{expect} 结尾且不含科学计数法）"
                    " —— 数字形态必须经 int64 归一化，字符串形态原样下发"
                )

    if errors:
        print(f"\n发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: chart 渲染、键引用与关键字段检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
