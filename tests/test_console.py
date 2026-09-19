"""校验终端真值展示、最终损失及失败时的有效样本口径。"""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_he_agent import AlternateClient

import he_agent as a
from models import ModelError, ModelResponse


class ConsoleTests(unittest.TestCase):
    def run_case(self, outputs, dry_run=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            rows = []
            for i in range(2):
                image = root / "images" / f"{i}.png"
                Image.new("L", (8, 8), color=i).save(image)
                rows.append(
                    {
                        "image": f"images/{i}.png",
                        "label": i,
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "source_dataset": "test",
                    }
                )
            (root / "labels/labels.txt").write_text(
                "".join(f"{r['image']} {r['label']}\n" for r in rows)
            )
            (root / "labels/manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            args = SimpleNamespace(
                dataset=root,
                output=root / "run",
                mode="baseline",
                split=None,
                prompt=None,
                limit=0,
                seed=42,
                model=None,
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=dry_run,
            )
            client = AlternateClient()
            capture = io.StringIO()
            with patch.object(client, "generate", side_effect=outputs) as generate:
                with redirect_stdout(capture):
                    code = a.run(args, client)
                if dry_run:
                    generate.assert_not_called()
                    self.assertFalse((args.output / "metrics.json").exists())
                    return code, capture.getvalue(), None, []
            saved = json.loads((args.output / "metrics.json").read_text())
            records = [
                json.loads(line)
                for line in (args.output / "predictions.jsonl").read_text().splitlines()
            ]
            return code, capture.getvalue(), saved, records

    def test_valid_progress_and_loss_match_saved_metrics(self):
        response = ModelResponse(text='{"p_he":0.8,"evidence":"visible"}', completed=True)
        code, text, saved, records = self.run_case([response, response])
        self.assertEqual(code, 0)
        for i, record in enumerate(records, 1):
            self.assertIn(f"{i}/2: true_label={record['true_label']}, p_he=0.8000", text)
        self.assertIn("predicted_label=1", text)
        self.assertIn("correct=True", text)
        self.assertIn("correct=False", text)
        self.assertIn(f"loss(log_loss)={saved['log_loss_on_valid']:.6f}", text)
        self.assertIn("valid=2/2, abstain=0, errors=0, unattempted=0", text)
        self.assertIn("Results saved to:", text)

    def test_abstention_and_error_have_no_fake_loss(self):
        response = ModelResponse(text='{"p_he":null,"evidence":"unclear"}', completed=True)
        code, text, saved, _ = self.run_case([response, ModelError("temporary")])
        self.assertEqual(code, 1)
        self.assertIn("status=abstain, predicted_label=null", text)
        self.assertIn("status=error, predicted_label=null", text)
        self.assertIn("loss(log_loss)=N/A, brier=N/A, valid=0/2, abstain=1, errors=1", text)
        self.assertIsNone(saved["log_loss_on_valid"])

    def test_fatal_error_reports_partial_completion(self):
        code, text, saved, _ = self.run_case([ModelError("no access", fatal=True)])
        self.assertEqual(code, 1)
        self.assertEqual(saved["unattempted"], 1)
        self.assertIn("errors=1, unattempted=1", text)
        self.assertIn("loss(log_loss)=N/A", text)

    def test_dry_run_does_not_report_loss(self):
        code, text, _, _ = self.run_case([], dry_run=True)
        self.assertEqual(code, 0)
        self.assertNotIn("loss(log_loss)", text)


if __name__ == "__main__":
    unittest.main()
