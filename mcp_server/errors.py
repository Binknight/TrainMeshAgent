"""JSON-RPC 与业务错误定义。"""

from typing import Any, Optional


class McpError(Exception):
    """可映射为 JSON-RPC error 的业务异常。"""

    def __init__(
        self,
        message: str,
        *,
        code: int = -32000,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data or {}


class UnknownToolError(McpError):
    def __init__(self, tool_name: str) -> None:
        super().__init__(f"unknown tool: {tool_name}", code=-32601)


class InvalidParamsError(McpError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=-32602)


class TaskNotFoundError(McpError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task_id not found: {task_id}", code=-32001)


class TaskNotReadyError(McpError):
    """任务未完成或结果尚未生成。"""

    def __init__(self, task_id: str, status: str) -> None:
        super().__init__(
            f"task {task_id} is not ready (status={status})",
            code=-32003,
        )


class NotImplementedToolError(McpError):
    """Tool 已注册但业务逻辑尚未实现。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"tool '{tool_name}' is registered but not yet implemented",
            code=-32002,
        )
