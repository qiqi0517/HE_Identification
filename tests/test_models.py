"""用模拟 HTTP 响应验证模型请求编码、重试与敏感信息处理。"""

import io
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import ImagePart, ModelError, ModelRequest, TextPart, create_model
from models.glm import GLMClient


class ModelTests(unittest.TestCase):
    def test_glm_wire_format_and_normalization(self):
        request = ModelRequest("task", (TextPart("question"), ImagePart(b"original", "image/png")))
        client = GLMClient()
        raw = {
            "choices": [
                {
                    "message": {
                        "content": '{"label":0,"evidence":"visible"}',
                        "reasoning_content": "hidden",
                    },
                    "finish_reason": "stop",
                }
            ],
            "id": "123",
            "model": "glm-5.3-flash",
            "usage": {"total_tokens": 12},
        }
        with patch.object(client, "_post", return_value=raw) as post:
            normalized = client.generate(request)
        payload = post.call_args.args[0]
        self.assertEqual(
            payload["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,b3JpZ2luYWw=",
        )
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertTrue(normalized.completed)
        self.assertEqual(normalized.response_id, "123")
        self.assertEqual(normalized.usage, {"total_tokens": 12})
        self.assertNotIn("hidden", repr(normalized))
        raw["choices"][0]["finish_reason"] = "length"
        self.assertFalse(client.normalize_response(raw).completed)
        with self.assertRaises(ModelError):
            client.normalize_response({})

    def test_registry_and_validation(self):
        self.assertEqual(create_model().model, "glm-5.3-flash")
        with self.assertRaises(ValueError):
            create_model("not-registered")
        with self.assertRaises(ValueError):
            create_model(model="text-only-model")
        with patch.dict(os.environ, {}, clear=True):
            client = create_model()
            with self.assertRaises(ValueError):
                client.validate_credentials()
            self.assertNotIn("api_key", client.describe())

    def test_retry_and_credential_redaction(self):
        client = GLMClient(retries=1)
        good = MagicMock()
        good.__enter__.return_value = io.StringIO('{"ok":true}')
        opener = MagicMock()
        opener.open.side_effect = [urllib.error.HTTPError("url", 429, "limited", {}, None), good]
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": "secret-placeholder"}),
            patch("models.glm.urllib.request.build_opener", return_value=opener),
            patch("models.glm.time.sleep"),
        ):
            self.assertEqual(client._post({}), {"ok": True})
        self.assertEqual(opener.open.call_count, 2)
        opener.reset_mock()
        opener.open.side_effect = urllib.error.HTTPError("url", 401, "secret-placeholder", {}, None)
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": "secret-placeholder"}),
            patch("models.glm.urllib.request.build_opener", return_value=opener),
        ):
            with self.assertRaises(ModelError) as ctx:
                client._post({})
        self.assertTrue(ctx.exception.fatal)
        self.assertNotIn("secret-placeholder", str(ctx.exception))
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
