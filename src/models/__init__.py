"""模型适配器注册入口；添加服务商时无需修改 Agent 的实验流程。

create_model 是唯一客户端工厂；新增服务商在 PROVIDERS 注册，主流程无需分支。
"""

from .base import ImagePart, ModelClient, ModelError, ModelRequest, ModelResponse, TextPart
from .deepseek import DeepSeekClient
from .glm import GLMClient
from .openai import OpenAIClient

PROVIDERS = {"glm": GLMClient, "openai": OpenAIClient, "deepseek": DeepSeekClient}


def create_model(
    provider="glm", model=None, max_tokens=8192, timeout=180, retries=2
) -> ModelClient:
    """根据注册名创建适配器；未知服务商直接报错，不自动回退。"""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}; available: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](model=model, max_tokens=max_tokens, timeout=timeout, retries=retries)


__all__ = [
    "ImagePart",
    "ModelClient",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "TextPart",
    "create_model",
]
