"""DeepSeek协议边界和网络错误测试；全部使用模拟响应，不产生费用。"""

import base64
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import he_agent
from models import ImagePart, ModelError, ModelRequest, TextPart, create_model
from models.deepseek import ENDPOINT, DeepSeekClient, NoRedirect


def response_body(probability=0.73):
    return {
        "id": "test-response",
        "model": "returned-model-id",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"p_he": probability, "evidence": "visible"}),
                    "reasoning_content": "private-reasoning-marker",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


class DeepSeekTests(unittest.TestCase):
    def test_registry_and_unsupported_models(self):
        self.assertIsInstance(create_model("deepseek"), DeepSeekClient)
        self.assertEqual(create_model("deepseek").model, "deepseek-flash")
        for model in ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro", "ds-v4.1-flash"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                create_model("deepseek", model=model)
        for args in (
            {"max_tokens": 0},
            {"max_tokens": 393217},
            {"timeout": float("nan")},
            {"timeout": 0},
            {"retries": -1},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                DeepSeekClient(**args)

    def test_original_images_order_and_parameters(self):
        request = ModelRequest(
            "unchanged system",
            (
                TextPart("reference label 1"),
                ImagePart(b"original-ref", "image/png"),
                TextPart("test image"),
                ImagePart(b"original-test", "image/jpeg"),
            ),
        )
        client = DeepSeekClient()
        with patch.object(client, "_post", return_value=response_body()) as post:
            result = client.generate(request)
        payload = post.call_args.args[0]
        self.assertEqual(
            payload["messages"][0], {"role": "system", "content": request.system_prompt}
        )
        parts = payload["messages"][1]["content"]
        for part, expected in zip([parts[1], parts[3]], [b"original-ref", b"original-test"]):
            self.assertEqual(base64.b64decode(part["image_url"]["url"].split(",")[1]), expected)
            self.assertEqual(part["image_url"]["detail"], "original")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["top_p"], 0.95)
        self.assertFalse(payload["stream"])
        self.assertEqual(he_agent.parse_prediction(result)["p_he"], 0.73)
        self.assertEqual(result.model, "returned-model-id")
        self.assertEqual(result.usage["prompt_cache_hit_tokens"], 80)
        self.assertNotIn("private-reasoning-marker", repr(result))

    def test_invalid_envelopes(self):
        for body in (
            None,
            [],
            {},
            {"choices": []},
            {"choices": [None]},
            {"choices": [{"message": {"content": []}}]},
            {"choices": response_body()["choices"] * 2},
        ):
            with self.subTest(body=body), self.assertRaises(ModelError):
                DeepSeekClient.normalize_response(body)

    def test_refusal_truncation_empty_and_tools_are_not_predictions(self):
        for reason in (
            "length",
            "content_filter",
            "tool_calls",
            "aborted",
            "insufficient_system_resource",
            None,
        ):
            raw = response_body()
            raw["choices"][0]["finish_reason"] = reason
            with self.subTest(reason=reason):
                self.assertFalse(DeepSeekClient.normalize_response(raw).completed)
        for fields in (
            {"content": None},
            {"content": "  "},
            {"refusal": "blocked"},
            {"tool_calls": [{"name": "unexpected"}]},
            {"role": "user"},
            {"function_call": {"name": "unexpected"}},
        ):
            raw = response_body()
            raw["choices"][0]["message"].update(fields)
            with self.subTest(fields=fields):
                response = DeepSeekClient.normalize_response(raw)
                self.assertFalse(response.completed)
                with self.assertRaises(ValueError):
                    he_agent.parse_prediction(response)
        parsed = he_agent.parse_prediction(DeepSeekClient.normalize_response(response_body(None)))
        self.assertIsNone(parsed["p_he"])

    def test_limits_do_not_silently_drop_or_resize_images(self):
        client = DeepSeekClient()
        for image in (ImagePart(b"", "image/png"), ImagePart(b"x", "image/tiff")):
            with self.assertRaises(ModelError):
                client.build_payload(ModelRequest("task", (image,)))
        request = ModelRequest("task", (ImagePart(b"1234", "image/png"),) * 2)
        for constant, limit in (
            ("MAX_IMAGES", 1),
            ("MAX_IMAGE_BYTES", 3),
            ("MAX_TOTAL_IMAGE_BYTES", 7),
        ):
            with patch("models.deepseek." + constant, limit), self.assertRaises(ModelError):
                client.build_payload(request)
        with (
            patch("models.deepseek.MAX_BODY_BYTES", 1),
            patch("models.deepseek.read_api_key", side_effect=AssertionError("key read")),
            self.assertRaises(ModelError),
        ):
            client._post({"large": "payload"})

    def test_http_retry_balance_errors_and_redaction(self):
        for status in (400, 401, 402, 403, 404, 422, 302, 429, 503):
            opener = MagicMock()
            # Use a fresh body on every attempt, as production closes errors after handling.
            opener.open.side_effect = lambda *a, code=status, **k: self.raise_http(code)
            with (
                self.subTest(status=status),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-test-secret"}),
                patch("models.deepseek.urllib.request.build_opener", return_value=opener),
                patch("models.deepseek.time.sleep"),
                self.assertRaises(ModelError) as ctx,
            ):
                DeepSeekClient(retries=1)._post({"model": "deepseek-flash"})
            temporary = status in (429, 503)
            self.assertEqual(opener.open.call_count, 2 if temporary else 1)
            self.assertEqual(ctx.exception.fatal, not temporary)
            self.assertNotIn("fake-test-secret", str(ctx.exception))
            self.assertNotIn("sensitive-response-body", str(ctx.exception))
            req = opener.open.call_args.args[0]
            self.assertEqual(req.full_url, ENDPOINT)
            self.assertEqual(req.get_header("Authorization"), "Bearer fake-test-secret")

    @staticmethod
    def raise_http(code):
        raise urllib.error.HTTPError(
            ENDPOINT, code, "fake-test-secret", {}, io.BytesIO(b"sensitive-response-body")
        )

    def test_network_retry_then_success_and_invalid_json(self):
        opener = MagicMock()
        good = MagicMock()
        good.__enter__.return_value = io.StringIO(json.dumps(response_body()))
        opener.open.side_effect = [urllib.error.URLError("fake-test-secret"), good]
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-test-secret"}),
            patch("models.deepseek.urllib.request.build_opener", return_value=opener),
            patch("models.deepseek.time.sleep"),
        ):
            result = DeepSeekClient(retries=1)._post({})
        self.assertEqual(result["model"], "returned-model-id")
        self.assertEqual(opener.open.call_count, 2)
        for bad in (urllib.error.URLError("fake-test-secret"), TimeoutError("fake-test-secret")):
            opener.open.side_effect = bad
            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-test-secret"}),
                patch("models.deepseek.urllib.request.build_opener", return_value=opener),
                self.assertRaises(ModelError) as ctx,
            ):
                DeepSeekClient(retries=0)._post({})
            self.assertNotIn("fake-test-secret", str(ctx.exception))
        good.__enter__.return_value = io.StringIO("fake-test-secret not JSON")
        opener.open.side_effect = None
        opener.open.return_value = good
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-test-secret"}),
            patch("models.deepseek.urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ModelError, "invalid JSON"),
        ):
            DeepSeekClient()._post({})

    def test_redirect_is_disabled(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other/"))


if __name__ == "__main__":
    unittest.main()
