"""智谱视觉模型适配器：负责鉴权、请求编码、重试和响应归一化。

调用链：generate → build_payload → _post → normalize_response。
适配器只翻译服务商协议；患病概率的解释和评分留给主流程。
"""

import base64
import http.client
import json
import time
import urllib.error
import urllib.request

from .base import ImagePart, ModelError, ModelRequest, ModelResponse, TextPart
from .credentials import read_api_key

ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5.3-flash"
VISION_MODELS = {"glm-5.3-flash", "glm-5.3-flashx"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止重定向，避免将鉴权请求转发到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """拒绝 urllib 的跳转请求，让鉴权头和原图只发往既定服务地址。"""
        return None


class GLMClient:
    """将统一多模态请求转换为智谱接口格式。"""

    def __init__(self, model=None, max_tokens=8192, timeout=180, retries=2):
        """校验模型白名单、输出预算和重试参数；此时不读密钥、不建立网络连接。"""
        self.model = model or DEFAULT_MODEL
        if self.model not in VISION_MODELS:
            raise ValueError(f"GLM adapter currently supports: {', '.join(sorted(VISION_MODELS))}")
        if max_tokens <= 0 or timeout <= 0 or retries < 0:
            raise ValueError("Invalid model call parameters")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def validate_credentials(self):
        """检查环境变量或项目本地密钥；不记录密钥，dry-run不调用此方法。"""
        read_api_key("glm")

    def describe(self):
        """返回可复现的模型参数；不包含 API Key。"""
        return {
            "provider": "glm",
            "model": self.model,
            "endpoint": ENDPOINT,
            "parameters": {
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout,
                "retries": self.retries,
                "temperature": 1,
                "top_p": 0.95,
                "thinking": {"type": "enabled"},
                "stream": False,
            },
        }

    def build_payload(self, request: ModelRequest):
        """保持片段顺序，将原始图像编码为 Data URL，不重新压缩图片。"""
        content = []
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:"
                            + part.mime
                            + ";base64,"
                            + base64.b64encode(part.data).decode("ascii")
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
            "stream": False,
            "temperature": 1,
            "top_p": 0.95,
            "thinking": {"type": "enabled"},
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def normalize_response(response):
        """只读取最终文本；仅正常 stop 视为完整回答。"""
        try:
            choice = response["choices"][0]
            text = choice["message"].get("content")
            reason = choice.get("finish_reason")
            if text is not None and not isinstance(text, str):
                raise TypeError("Non-text final answer")
            return ModelResponse(
                text=text,
                completed=reason == "stop",
                finish_reason=reason,
                response_id=response.get("id"),
                model=response.get("model"),
                usage=response.get("usage"),
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ModelError("Malformed GLM response envelope") from None

    def _post(self, payload):
        """发送请求并有限重试临时错误；永久性调用错误标记为 fatal。"""
        key = read_api_key("glm")
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(NoRedirect())
        for attempt in range(self.retries + 1):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or 500 <= e.code < 600
                if retryable and attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    f"HTTP {e.code}; check model access, API balance, parameters or rate limit",
                    fatal=e.code in {400, 401, 402, 403, 404},
                ) from None
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    "Network error or timeout; request may already have been billed"
                ) from None
            except (ValueError, UnicodeError):
                raise ModelError("GLM returned invalid JSON") from None
        raise ModelError("No response")

    def generate(self, request: ModelRequest) -> ModelResponse:
        """完成请求编码、接口调用及响应归一化。"""
        return self.normalize_response(self._post(self.build_payload(request)))
