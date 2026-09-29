"""OpenAI Responses API 适配器；仅处理传输协议，不解释医学诊断。

调用链：generate → build_payload → _post → normalize_response。
Responses 的 output 可能含 reasoning／refusal，只收集完整 assistant 最终文本。
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

ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-terra"
VISION_MODELS = {DEFAULT_MODEL}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """不向重定向目标转发带鉴权信息的请求。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """返回 None 禁止自动跳转；鉴权信息不能随重定向转发。"""
        return None


class OpenAIClient:
    """用 GPT-5.6 Terra 进行独立图像推理，返回统一 ModelResponse。"""

    def __init__(self, model=None, max_tokens=8192, timeout=180, retries=2):
        """固定受支持模型并校验预算、超时和重试配置；只建立客户端，不调用 API。"""
        self.model = model or DEFAULT_MODEL
        if self.model not in VISION_MODELS:
            raise ValueError(
                f"OpenAI adapter currently supports: {', '.join(sorted(VISION_MODELS))}"
            )
        if (
            type(max_tokens) is not int
            or max_tokens <= 0
            or type(retries) is not int
            or retries < 0
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid model call parameters")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def validate_credentials(self):
        """检查环境变量或项目本地密钥；不记录密钥，dry-run不调用此方法。"""
        read_api_key("openai")

    def describe(self):
        """记录实际请求参数，便于比较服务商差异，不包含凭据。"""
        return {
            "provider": "openai",
            "model": self.model,
            "endpoint": ENDPOINT,
            "parameters": {
                "max_output_tokens": self.max_tokens,
                "timeout_seconds": self.timeout,
                "retries": self.retries,
                "reasoning": {"effort": "medium"},
                "image_detail": "original",
                "store": False,
                "stream": False,
                "truncation": "disabled",
            },
        }

    def build_payload(self, request: ModelRequest):
        """保持参考图、标签和测试图的顺序；不裁剪、缩放或重新编码图像。"""
        content = []
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append(
                    {
                        "type": "input_image",
                        "detail": "original",
                        "image_url": f"data:{part.mime};base64,"
                        + base64.b64encode(part.data).decode("ascii"),
                    }
                )
            else:
                raise ValueError("Unsupported content part")
        return {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": self.max_tokens,
            "store": False,
            "stream": False,
            "truncation": "disabled",
        }

    @staticmethod
    def normalize_response(response):
        """收集 assistant 最终文本；拒绝、工具输出或未完成响应不能成为有效预测。"""
        try:
            status = response["status"]
            if status not in {
                "completed",
                "incomplete",
                "failed",
                "cancelled",
                "queued",
                "in_progress",
            }:
                raise ValueError("Unknown response status")
            output = response["output"]
            if not isinstance(output, list):
                raise TypeError("Expected output items")
            texts = []
            refusal = False
            unexpected = False
            unfinished = False
            for item in output:
                item_type = item["type"]
                # 不记录 reasoning item 或其中的 summary；它们不是最终回答。
                if item_type == "reasoning":
                    continue
                if item_type != "message":
                    unexpected = True
                    continue
                if item["role"] != "assistant":
                    unexpected = True
                    continue
                unfinished |= item["status"] != "completed"
                if not isinstance(item["content"], list):
                    raise TypeError("Expected message content")
                for part in item["content"]:
                    if part["type"] == "output_text":
                        if not isinstance(part["text"], str):
                            raise TypeError("Non-text final answer")
                        texts.append(part["text"])
                    elif part["type"] == "refusal":
                        refusal = True
                    else:
                        unexpected = True
            text = "\n".join(texts) if texts else None
            completed = (
                status == "completed"
                and not (refusal or unexpected or unfinished)
                and bool(text and text.strip())
            )
            reason = status
            if status == "incomplete":
                details = response.get("incomplete_details") or {}
                # 仅保留约定原因，避免把任意服务端文本带入错误日志。
                if details.get("reason") in {"max_output_tokens", "content_filter"}:
                    reason = details["reason"]
            if refusal:
                reason = "refusal"
            elif unexpected:
                reason = "unexpected_output"
            elif unfinished:
                reason = "incomplete_message"
            elif status == "completed" and not completed:
                reason = "empty_output"
            return ModelResponse(
                text=text,
                completed=bool(completed),
                finish_reason=reason,
                response_id=response.get("id"),
                model=response.get("model"),
                usage=response.get("usage"),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ModelError("Malformed OpenAI response envelope") from None

    @staticmethod
    def _quota_exhausted(error):
        """区分永久额度不足与临时限流；错误正文只在内存中检查，不写日志。"""
        try:
            body = json.loads(error.read(65536))
            details = body.get("error", {})
            return (
                details.get("code") in {"insufficient_quota", "billing_hard_limit_reached"}
                or details.get("type") == "insufficient_quota"
            )
        except (ValueError, TypeError, AttributeError, OSError):
            return False

    def _post(self, payload):
        """发送到官方固定地址；仅对临时错误有限重试，不回显凭据或图片。"""
        key = read_api_key("openai")
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(NoRedirect())
        for attempt in range(self.retries + 1):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                quota_exhausted = error.code == 429 and self._quota_exhausted(error)
                retryable = (error.code == 429 or 500 <= error.code < 600) and not quota_exhausted
                if retryable and attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    "OpenAI quota exhausted; check API billing"
                    if quota_exhausted
                    else f"OpenAI HTTP {error.code}; check access, parameters or rate limit",
                    fatal=quota_exhausted or error.code in {400, 401, 402, 403, 404},
                ) from None
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(
                    "OpenAI network error or timeout; request may already have been billed"
                ) from None
            except (ValueError, UnicodeError):
                raise ModelError("OpenAI returned invalid JSON") from None
        raise ModelError("No OpenAI response")

    def generate(self, request: ModelRequest) -> ModelResponse:
        """编码请求、发送到 Responses API，再转换为统一响应。"""
        return self.normalize_response(self._post(self.build_payload(request)))
