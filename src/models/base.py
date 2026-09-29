"""定义与服务商无关的多模态请求、响应和模型客户端接口。

请求／响应数据类定义模型无关的数据边界；ModelClient 是接口契约，
ModelError 区分可继续记录的单图失败与需要停止批量调用的致命错误。
"""

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
    # completed 只表示正常完成，不包括截断、API 拒绝或工具调用。
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
        """保存已脱敏的错误说明和批量终止标记，由主流程决定是否继续下一张。"""
        super().__init__(message)
        self.fatal = fatal


class ModelClient(Protocol):
    """所有模型适配器遵守的接口，Agent 只依赖此契约。"""

    def validate_credentials(self) -> None:
        """在真实调用前核验凭据；缺失时提早失败，不创建无用的付费请求。"""
        ...

    def describe(self) -> dict[str, Any]:
        """返回可复现的配置元数据，不含密钥或其他凭据。"""
        ...

    def generate(self, request: ModelRequest) -> ModelResponse:
        """编码并调用服务商，将响应转换成统一的最终回答对象。"""
        ...
