"""离线验证 Agent 请求隔离、概率解析和模型适配器注入。"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import he_agent as a
from models import ImagePart, ModelResponse, TextPart


class AlternateClient:
    """Different provider, with no GLM request/response or credential dependency."""

    def __init__(self):
        self.requests = []
        self.credential_checks = 0

    def describe(self):
        return {"provider": "alternate-test", "model": "vision-test", "parameters": {}}

    def validate_credentials(self):
        self.credential_checks += 1

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            text='{"p_he":0.8,"evidence":"visible features"}',
            completed=True,
            response_id="fake-1",
            model="vision-test",
            finish_reason="complete",
        )


def response(probability, completed=True):
    return ModelResponse(
        text=json.dumps({"p_he": probability, "evidence": "visible features"}), completed=completed
    )


class Tests(unittest.TestCase):
    def test_request_has_no_ground_truth_or_provider_fields(self):
        r = a.build_request(b"image", "image/png", "task")
        self.assertEqual(r.system_prompt, "task")
        self.assertEqual(len(r.content), 2)
        self.assertIsInstance(r.content[0], TextPart)
        self.assertEqual(r.content[1], ImagePart(data=b"image", mime="image/png"))
        self.assertNotIn("label", r.content[0].text)
        self.assertNotIn("model", vars(r))

    def test_invalid_bool_and_truncated_answer(self):
        with self.assertRaises(ValueError):
            a.parse_prediction(response(True))
        with self.assertRaises(ValueError):
            a.parse_prediction(response(0, completed=False))
        self.assertIsNone(a.parse_prediction(response(None))["p_he"])

    def test_failure_is_not_negative(self):
        rows = [
            {"status": "ok", "true_label": 1, "p_he": 0.8},
            {"status": "error", "true_label": 0, "p_he": None},
            {"status": "abstain", "true_label": 0, "p_he": None},
        ]
        m = a.metrics(rows, 4)
        self.assertEqual(m["coverage"], 0.25)
        self.assertIsNone(m["specificity_on_valid"])
        self.assertEqual(m["unattempted"], 1)

    def test_alternate_provider_full_workflow_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            f = root / "images/test.png"
            Image.new("L", (16, 16)).save(f)
            (root / "labels/labels.txt").write_text("images/test.png 1\n")
            (root / "labels/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "image": "images/test.png",
                        "label": 1,
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                        "source_dataset": "test",
                    }
                )
                + "\n"
            )
            prompt = root / "prompt.txt"
            prompt.write_text("estimate p_he")
            args = SimpleNamespace(
                dataset=root,
                output=root / "run",
                prompt=prompt,
                limit=0,
                seed=42,
                model=None,
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=False,
            )
            client = AlternateClient()
            with patch.object(
                a, "create_model", side_effect=AssertionError("Must use injected client")
            ):
                self.assertEqual(a.run(args, model_client=client), 0)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(client.credential_checks, 1)
            self.assertEqual(
                json.loads((root / "run/metrics.json").read_text())["accuracy_on_valid"], 1
            )
            result = json.loads((root / "run/predictions.jsonl").read_text())
            self.assertEqual(result["p_he"], 0.8)
            self.assertAlmostEqual(result["p_non_he"], 0.2)
            self.assertEqual(result["predicted_label"], 1)
            self.assertEqual(result["output_schema"], "he-probability-v2")
            config = json.loads((root / "run/run.json").read_text())
            self.assertEqual(config["provider"], "alternate-test")
            self.assertEqual(config["model"], "vision-test")
            args.output = root / "dry"
            args.dry_run = True
            self.assertEqual(a.run(args, model_client=client), 0)
            self.assertEqual(client.credential_checks, 1)
            self.assertEqual(len(client.requests), 1)
            f.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                a.load_samples(root)


if __name__ == "__main__":
    unittest.main()
