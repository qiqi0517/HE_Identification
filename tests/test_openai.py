"""离线核对 OpenAI Responses 协议、异常处理和参数隔离。"""

import copy
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from he_agent import parse_prediction
from models import ImagePart, ModelError, ModelRequest, TextPart, create_model
from models.openai import ENDPOINT, NoRedirect, OpenAIClient


def response_body(probability=0.73):
    """构造含 reasoning item 的服务端响应，确保适配器只取最终回答。"""
    return {
        "id": "resp-test",
        "model": "gpt-5.6-terra",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"text": "hidden-reasoning"}]},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"p_he": probability, "evidence": "visible structures"}),
                    }
                ],
            },
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 80,
            "total_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 50},
        },
    }


class OpenAITests(unittest.TestCase):
    def test_request_encoding_order_and_parameters(self):
        client = OpenAIClient(max_tokens=2048)
        request = ModelRequest(
            "medical prompt",
            (
                TextPart("reference label: 1"),
                ImagePart(b"reference", "image/png"),
                TextPart("test image"),
                ImagePart(b"original", "image/jpeg"),
            ),
        )
        with patch.object(client, "_post", return_value=response_body()) as post:
            response = client.generate(request)
        payload = post.call_args.args[0]
        self.assertEqual(payload["instructions"], request.system_prompt)
        parts = payload["input"][0]["content"]
        self.assertEqual(
            [p["type"] for p in parts], ["input_text", "input_image", "input_text", "input_image"]
        )
        self.assertEqual(parts[-1]["image_url"], "data:image/jpeg;base64,b3JpZ2luYWw=")
        self.assertEqual(parts[1]["detail"], "original")
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(payload["max_output_tokens"], 2048)
        self.assertFalse(payload["store"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["truncation"], "disabled")
        for key in ("temperature", "top_p", "thinking", "previous_response_id", "tools"):
            self.assertNotIn(key, payload)
        self.assertEqual(parse_prediction(response)["p_he"], 0.73)
        self.assertNotIn("hidden-reasoning", repr(response))
        self.assertEqual(response.usage["total_tokens"], 200)

    def test_registry_credentials_and_validation(self):
        self.assertEqual(create_model("openai").model, "gpt-5.6-terra")
        self.assertEqual(create_model().model, "glm-5.3-flash")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("models.credentials.load_local_keys", return_value={}),
        ):
            with self.assertRaises(ValueError):
                OpenAIClient().validate_credentials()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-placeholder"}):
            OpenAIClient().validate_credentials()
            self.assertNotIn("private-placeholder", json.dumps(OpenAIClient().describe()))
        for kwargs in (
            {"model": "gpt-image-2"},
            {"timeout": float("nan")},
            {"max_tokens": 0},
            {"retries": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenAIClient(**kwargs)
        with self.assertRaises(ValueError):
            OpenAIClient().build_payload(ModelRequest("task", (object(),)))

    def test_incomplete_and_failed_answers_never_score(self):
        for status in ("incomplete", "failed", "cancelled", "queued", "in_progress"):
            body = response_body()
            body["status"] = status
            body["incomplete_details"] = {"reason": "max_output_tokens"}
            with self.subTest(status=status):
                response = OpenAIClient.normalize_response(body)
                self.assertFalse(response.completed)
                with self.assertRaises(ValueError):
                    parse_prediction(response)
        self.assertEqual(
            OpenAIClient.normalize_response(
                {
                    **response_body(),
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            ).finish_reason,
            "max_output_tokens",
        )

    def test_refusal_empty_tools_and_partial_messages(self):
        body = response_body()
        body["output"][1]["content"].append({"type": "refusal", "refusal": "cannot assess"})
        variants = [body, {**response_body(), "output": []}]
        tool = response_body()
        tool["output"].append({"type": "function_call"})
        variants.append(tool)
        partial = response_body()
        partial["output"][1]["status"] = "in_progress"
        variants.append(partial)
        for body in variants:
            with self.subTest(output=body["output"]):
                response = OpenAIClient.normalize_response(body)
                self.assertFalse(response.completed)
                with self.assertRaises(ValueError):
                    parse_prediction(response)
        self.assertEqual(OpenAIClient.normalize_response(variants[0]).finish_reason, "refusal")

    def test_split_text_and_null_probability(self):
        body = response_body(None)
        text = body["output"][1]["content"][0]["text"]
        index = text.index(",") + 1
        body["output"][1]["content"] = [
            {"type": "output_text", "text": text[:index]},
            {"type": "output_text", "text": text[index:]},
        ]
        self.assertIsNone(parse_prediction(OpenAIClient.normalize_response(body))["p_he"])

    def test_malformed_envelope_is_sanitized(self):
        bad = response_body()
        bad["output"][1]["content"][0]["text"] = 42
        for body in (None, {}, [], {"status": "completed", "output": "invalid"}, bad):
            with self.subTest(body=body), self.assertRaisesRegex(ModelError, "Malformed OpenAI"):
                OpenAIClient.normalize_response(copy.deepcopy(body))

    def test_http_request_and_temporary_retry(self):
        client = OpenAIClient(retries=1)
        good = MagicMock()
        good.__enter__.return_value = io.StringIO('{"ok":true}')
        opener = MagicMock()
        opener.open.side_effect = [
            urllib.error.HTTPError("url", 429, "limited", {}, io.BytesIO(b"{}")),
            good,
        ]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "private-placeholder"}),
            patch("models.openai.urllib.request.build_opener", return_value=opener),
            patch("models.openai.time.sleep") as sleep,
        ):
            self.assertEqual(client._post({"input": []}), {"ok": True})
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(1)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer private-placeholder")
        self.assertEqual(request.get_method(), "POST")
        self.assertIsNone(
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid")
        )

    def test_permanent_and_quota_errors_stop_without_leaks(self):
        for code in (400, 401, 403, 404, 429):
            body = io.BytesIO(
                b'{"error":{"code":"insufficient_quota","message":"private-placeholder"}}'
            )
            opener = MagicMock()
            opener.open.side_effect = urllib.error.HTTPError(
                "url", code, "private-placeholder", {}, body
            )
            with (
                self.subTest(code=code),
                patch.dict(os.environ, {"OPENAI_API_KEY": "private-placeholder"}),
                patch("models.openai.urllib.request.build_opener", return_value=opener),
                patch("models.openai.time.sleep") as sleep,
            ):
                with self.assertRaises(ModelError) as caught:
                    OpenAIClient()._post({})
                self.assertTrue(caught.exception.fatal)
                self.assertNotIn("private-placeholder", str(caught.exception))
                self.assertEqual(opener.open.call_count, 1)
                sleep.assert_not_called()

    def test_network_and_server_retries_are_bounded(self):
        for error in (
            urllib.error.URLError("private-placeholder"),
            TimeoutError("private-placeholder"),
            urllib.error.HTTPError("url", 503, "private-placeholder", {}, None),
        ):
            opener = MagicMock()
            opener.open.side_effect = error
            with (
                self.subTest(error=type(error)),
                patch.dict(os.environ, {"OPENAI_API_KEY": "private-placeholder"}),
                patch("models.openai.urllib.request.build_opener", return_value=opener),
                patch("models.openai.time.sleep"),
            ):
                with self.assertRaises(ModelError) as caught:
                    OpenAIClient(retries=2)._post({})
                self.assertFalse(caught.exception.fatal)
                self.assertNotIn("private-placeholder", str(caught.exception))
                self.assertEqual(opener.open.call_count, 3)

    def test_invalid_json_does_not_retry(self):
        good = MagicMock()
        good.__enter__.return_value = io.StringIO("not json")
        opener = MagicMock()
        opener.open.return_value = good
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "private-placeholder"}),
            patch("models.openai.urllib.request.build_opener", return_value=opener),
        ):
            with self.assertRaisesRegex(ModelError, "invalid JSON"):
                OpenAIClient()._post({})
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
