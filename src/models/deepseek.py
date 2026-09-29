"""DeepSeek视觉适配器：原图直传、独立鉴权及最终回答归一化。

调用链：generate → build_payload → _post → normalize_response。
额外校验单图、总图和编码请求体上限；超限时报错，不降质压图。
"""

import base64
import http.client
import json
import math
import time
import urllib.error
import urllib.request

from .base import ImagePart, ModelError, ModelRequest, ModelResponse, TextPart
from .credentials import read_api_key

ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
# 只开放已核对官方视觉文档的名称，不把文本模型或退役别名静默映射过来。
VISION_MODELS = {DEFAULT_MODEL}
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024
MAX_BODY_BYTES = 48 * 1024 * 1024
MAX_IMAGES = 600
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """不向重定向目标转发鉴权或超声图像。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """拒绝跳转，避免携带密钥与图片访问返回的其他 URL。"""
        return None


class DeepSeekClient:
    """实现主Agent已有契约，不改变提示词、参考选择或概率评分。"""

    def __init__(self, model=None, max_tokens=8192, timeout=180, retries=2):
        """校验视觉模型名称、有限超时、整数 token 上限和非负重试次数。"""
        self.model = model or DEFAULT_MODEL
        if self.model not in VISION_MODELS:
            raise ValueError("DeepSeek vision adapter supports only deepseek-flash")
        if (
            type(max_tokens) is not int
            or not 1 <= max_tokens <= 393216
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
            or type(retries) is not int
            or retries < 0
        ):
            raise ValueError("Invalid DeepSeek call parameters")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def validate_credentials(self):
        """真实运行前检查key；dry-run不会读取key，也不会发送请求。"""
        read_api_key("deepseek")

    def describe(self):
        """保存显式调用参数；别名对应的实际返回模型另随每次响应记录。"""
        return {
            "provider": "deepseek",
            "model": self.model,
            "endpoint": ENDPOINT,
            "parameters": {
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout,
                "retries": self.retries,
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "top_p": 0.95,
                "detail": "original",
                "stream": False,
            },
        }

    def build_payload(self, request: ModelRequest):
        """保留图片字节和顺序；超限报错，不自动压缩或丢弃参考图。"""
        content = []
        count = total = 0
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.mime not in IMAGE_MIMES:
                    raise ModelError("Unsupported DeepSeek image MIME type", fatal=True)
                size = len(part.data)
                count += 1
                total += size
                if (
                    not size
                    or size > MAX_IMAGE_BYTES
                    or count > MAX_IMAGES
                    or total > MAX_TOTAL_IMAGE_BYTES
                ):
                    raise ModelError(
                        "DeepSeek inline image limits exceeded or empty image", fatal=True
                    )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:"
                            + part.mime
                            + ";base64,"
                            + base64.b64encode(part.data).decode("ascii"),
                            "detail": "original",
                        },
                    }
                )
            else:
                raise ValueError("Unsupported content part")
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "top_p": 0.95,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    @staticmethod
    def normalize_response(response):
        """忽略reasoning_content，只接收正常结束的单个assistant最终回答。"""
        try:
            choices = response["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Expected one choice")
            choice = choices[0]
            message = choice["message"]
            text = message.get("content")
            reason = choice.get("finish_reason")
            if text is not None and not isinstance(text, str):
                raise TypeError("Non-text answer")
            complete = (
                reason == "stop"
                and isinstance(text, str)
                and bool(text.strip())
                and message.get("role", "assistant") == "assistant"
                and not message.get("refusal")
                and not message.get("tool_calls")
                and not message.get("function_call")
            )
            return ModelResponse(
                text=text,
                completed=complete,
                finish_reason=reason,
                response_id=response.get("id"),
                model=response.get("model"),
                usage=response.get("usage"),
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelError("Malformed DeepSeek response envelope") from None

    def _post(self, payload):
        """仅重试临时网络/服务错误；余额或鉴权错误停止，错误正文不入日志。"""
        body = json.dumps(payload).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise ModelError("DeepSeek request exceeds 48 MiB; images were not resized", fatal=True)
        key = read_api_key("deepseek")
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(NoRedirect())
        for attempt in range(self.retries + 1):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                retryable = code in {408, 429} or 500 <= code < 600
                if retryable and attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    f"DeepSeek HTTP {code}; check API key, balance, model access or request limits",
                    fatal=not retryable,
                ) from None
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    "DeepSeek network error or timeout; request may already have been billed"
                ) from None
            except (ValueError, UnicodeError):
                raise ModelError("DeepSeek returned invalid JSON") from None
        raise ModelError("No DeepSeek response")

    def generate(self, request: ModelRequest) -> ModelResponse:
        """调用官方多模态接口；不经过GLM描述图片或其他文本代理。"""
        return self.normalize_response(self._post(self.build_payload(request)))
