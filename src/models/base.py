"""定义与服务商无关的多模态请求、响应和模型客户端接口。"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TextPart:
    """请求中的文字片段，包括任务说明或允许使用的参考标签。"""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """原始图片字节与 MIME 类型；repr 隐藏字节，避免日志输出图像内容。"""

    data: bytes = field(repr=False)
    mime: str


@dataclass(frozen=True)
class ModelRequest:
    """一次独立推理的系统提示词与有序多模态片段。"""

    system_prompt: str
    content: tuple[TextPart | ImagePart, ...]


@dataclass(frozen=True)
class ModelResponse:
    # completed means a normal end, not truncation, refusal, or tool invocation.
    """归一化的最终回答及运行元数据，不包含模型隐藏推理。"""

    text: str | None
    completed: bool
    response_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class ModelError(RuntimeError):
    """模型调用异常；fatal=True 表示应停止后续批量调用。"""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class ModelClient(Protocol):
    """所有模型适配器遵守的接口，Agent 只依赖此契约。"""

    def validate_credentials(self) -> None:
        """Fail before a live run if required credentials are unavailable."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return reproducibility metadata, never keys or other secrets."""
        ...

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Encode, call, and normalize this provider's response."""
        ...
