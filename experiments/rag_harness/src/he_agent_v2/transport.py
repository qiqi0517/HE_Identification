"""统计既有 urllib 适配器的真实 HTTP 尝试，且不读取请求头或图像正文。"""

import contextvars
import sys
import time

from models import ModelError

_ACTIVE = contextvars.ContextVar("he_v2_http_counter", default=None)


def _audit(event, args):
    # urllib 每次 opener.open() 发出一次审计事件，重试也会计数。
    # 不访问 args：其中可能包含 Authorization 和 Base64 原图。
    """通过 urllib 审计事件计数实际 HTTP 尝试，不读取包含密钥的事件参数。"""
    if event == "urllib.Request":
        state = _ACTIVE.get()
        if state is not None:
            if state["http_attempts"] >= state["http_limit"]:
                raise ModelError("HTTP attempt budget exhausted", fatal=True)
            state["http_attempts"] += 1


sys.addaudithook(_audit)


class MeteredClient:
    """两次逻辑调用与单阶段最多三次 HTTP 尝试分别受控。"""

    def __init__(self, client, retries=2):
        """保存实际客户端与每阶段重试预算，初始化逻辑调用记录。"""
        if not 0 <= retries <= 2:
            raise ValueError("At most two retries per phase")
        self.client = client
        self.retries = retries
        self.calls = []

    def generate(self, request):
        """限制每图最多两次逻辑调用，并以局部上下文分别记录 HTTP 重试与用量。"""
        if len(self.calls) >= 2:
            raise ModelError("Logical call budget exhausted", fatal=True)
        state = {
            "logical_call": len(self.calls) + 1,
            "http_attempts": 0,
            "http_limit": self.retries + 1,
            "usage": None,
        }
        self.calls.append(state)
        token = _ACTIVE.set(state)
        start = time.perf_counter()
        try:
            response = self.client.generate(request)
            state["usage"] = response.usage
            return response
        finally:
            state["latency_seconds"] = time.perf_counter() - start
            _ACTIVE.reset(token)
