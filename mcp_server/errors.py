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


class QueueFullError(McpError):
    """排队任务已达上限（背压）：调用方应稍后重试，不得静默入队。

    JSON-RPC code 固定 **-32004**（与 TaskNotReadyError 的 -32003 区分开：
    一个是「资源/队列满，稍后重试」，一个是「任务结果未就绪」，调用方按 code
    即可判定是否走退避重试）。
    """

    def __init__(self, *, queue_size: int, max_queue_size: int) -> None:
        super().__init__(
            f"队列已满：排队任务已达上限 max_queue_size={max_queue_size}"
            f"（当前排队 {queue_size} 个，执行槽位已占满），请稍后重试",
            code=-32004,
            data={
                "reason": "queue_full",
                "queue_size": queue_size,
                "max_queue_size": max_queue_size,
            },
        )
        self.queue_size = queue_size
        self.max_queue_size = max_queue_size


class NotImplementedToolError(McpError):
    """Tool 已注册但业务逻辑尚未实现。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"tool '{tool_name}' is registered but not yet implemented",
            code=-32002,
        )
