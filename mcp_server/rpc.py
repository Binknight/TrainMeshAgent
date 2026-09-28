"""JSON-RPC 2.0 请求/响应模型（§2.1）。"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: Optional[dict[str, Any]] = None
    id: Optional[Union[int, str]] = None


class ToolsCallParams(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Optional[dict[str, Any]] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    result: Optional[dict[str, Any]] = None
    error: Optional[JsonRpcError] = None
    id: Optional[Union[int, str]] = None

    def model_dump_response(self) -> dict[str, Any]:
        """保证 result 字段始终存在（§13）。"""
        data = self.model_dump(exclude_none=True)
        if "result" not in data:
            data["result"] = None
        return data
