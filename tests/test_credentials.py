"""本地密钥读取优先级、HTTP鉴权与无密钥离线行为。只使用虚构测试密钥。"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_experiment_matrix as fixtures

import he_agent
from models import create_model, credentials


class CredentialTests(unittest.TestCase):
    def test_file_fallback_and_no_environment_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "keys.json"
            path.write_text(
                json.dumps(
                    {
                        "ZHIPU_API_KEY": "  local-glm-test  ",
                        "OPENAI_API_KEY": "local-openai-test",
                        "DEEPSEEK_API_KEY": "local-deepseek-test",
                    }
                )
            )
            with (
                patch.object(credentials, "LOCAL_KEYS_PATH", path),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(credentials.read_api_key("glm"), "local-glm-test")
                self.assertEqual(credentials.read_api_key("openai"), "local-openai-test")
                self.assertEqual(credentials.read_api_key("deepseek"), "local-deepseek-test")
                self.assertNotIn("ZHIPU_API_KEY", os.environ)
                self.assertNotIn("OPENAI_API_KEY", os.environ)
                with patch.dict(os.environ, {"ZHIPU_API_KEY": "  "}):
                    self.assertEqual(credentials.read_api_key("glm"), "local-glm-test")

    def test_nonempty_environment_precedes_file(self):
        for provider, name in credentials.KEY_NAMES.items():
            with (
                self.subTest(provider=provider),
                patch.dict(os.environ, {name: "env-test-key"}, clear=True),
                patch.object(
                    credentials, "load_local_keys", side_effect=AssertionError("file read")
                ),
            ):
                self.assertEqual(credentials.read_api_key(provider), "env-test-key")

    def test_missing_empty_wrong_type_and_invalid_format(self):
        for value in ("", None, 123, "test key", "test\nkey", "测试密钥"):
            with (
                self.subTest(value_type=type(value).__name__),
                patch.dict(os.environ, {}, clear=True),
                patch.object(credentials, "load_local_keys", return_value={"ZHIPU_API_KEY": value}),
            ):
                with self.assertRaises(ValueError) as ctx:
                    credentials.read_api_key("glm")
                if isinstance(value, str) and len(value) > 3:
                    self.assertNotIn(value, str(ctx.exception))
        with tempfile.TemporaryDirectory() as d:
            with patch.object(credentials, "LOCAL_KEYS_PATH", Path(d) / "missing.json"):
                self.assertEqual(credentials.load_local_keys(), {})

    def test_malformed_json_and_io_errors_are_sanitized(self):
        for raw in (
            "secret-test-fragment {",
            "[]",
            '{"ZHIPU_API_KEY":"secret-test-fragment","ZHIPU_API_KEY":"x"}',
        ):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "keys.json"
                path.write_text(raw)
                with patch.object(credentials, "LOCAL_KEYS_PATH", path):
                    with self.assertRaises(ValueError) as ctx:
                        credentials.load_local_keys()
                    self.assertNotIn("secret-test-fragment", str(ctx.exception))
        with patch.object(Path, "read_text", side_effect=PermissionError("secret-test-fragment")):
            with self.assertRaises(ValueError) as ctx:
                credentials.load_local_keys()
            self.assertNotIn("secret-test-fragment", str(ctx.exception))

    def test_local_keys_reach_all_adapters_without_metadata_leak(self):
        for provider, name in credentials.KEY_NAMES.items():
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "keys.json"
                path.write_text(json.dumps({name: "fake-local-test-key"}))
                client = create_model(provider)
                good = MagicMock()
                good.__enter__.return_value = io.StringIO('{"ok":true}')
                opener = MagicMock()
                opener.open.return_value = good
                with (
                    patch.object(credentials, "LOCAL_KEYS_PATH", path),
                    patch.dict(os.environ, {}, clear=True),
                    patch(f"models.{provider}.urllib.request.build_opener", return_value=opener),
                ):
                    client.validate_credentials()
                    self.assertEqual(client._post({}), {"ok": True})
                request = opener.open.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer fake-local-test-key")
                self.assertNotIn("fake-local-test-key", json.dumps(client.describe()))
                self.assertNotIn("fake-local-test-key", request.data.decode())
                self.assertNotIn("fake-local-test-key", repr(vars(client)))

    def test_live_run_does_not_save_local_key(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, split = fixtures.MatrixTests().fixture(root)
            keyfile = root / "api_keys.local.json"
            keyfile.write_text('{"ZHIPU_API_KEY":"fake-local-test-key"}')
            args = SimpleNamespace(
                dataset=root,
                split=split,
                mode="reference",
                prompt=None,
                provider="glm",
                model=None,
                output=root / "output",
                limit=1,
                seed=42,
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=False,
                review=True,
                review_threshold=0.2,
            )
            client = create_model("glm")
            body = {
                "choices": [
                    {
                        "message": {"content": '{"p_he":0.8,"evidence":"visible"}'},
                        "finish_reason": "stop",
                    }
                ]
            }
            capture = io.StringIO()
            with (
                patch.object(credentials, "LOCAL_KEYS_PATH", keyfile),
                patch.dict(os.environ, {}, clear=True),
                patch.object(client, "_post", return_value=body),
                redirect_stdout(capture),
            ):
                self.assertEqual(he_agent.run_probability(args, client), 0)
            self.assertNotIn("fake-local-test-key", capture.getvalue())
            for file in args.output.rglob("*"):
                if file.is_file():
                    self.assertNotIn("fake-local-test-key", file.read_text())

    def test_dry_run_does_not_read_local_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, split = fixtures.MatrixTests().fixture(root)
            args = SimpleNamespace(
                dataset=root,
                split=split,
                mode="knowledge",
                prompt=None,
                provider="openai",
                model=None,
                output=root / "output",
                limit=1,
                seed=42,
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=True,
            )
            client = create_model("openai")
            with (
                patch.object(
                    credentials, "load_local_keys", side_effect=AssertionError("key file read")
                ),
                patch.object(client, "generate", side_effect=AssertionError("API call")),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(he_agent.run_probability(args, client), 0)


if __name__ == "__main__":
    unittest.main()
