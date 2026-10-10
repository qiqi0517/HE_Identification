"""核对连续概率协议、手算评分及阈值分类的边界。"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import he_agent as a
from models import ModelResponse


def parse(text):
    return a.parse_prediction(ModelResponse(text=text, completed=True))


def row(label, probability):
    return {"status": "ok", "true_label": label, "p_he": probability}


class ProbabilityTests(unittest.TestCase):
    def test_valid_continuous_probabilities_and_null(self):
        for p in (0, 0.0137, 0.5, 0.825, 1, None):
            with self.subTest(p=p):
                self.assertEqual(parse(json.dumps({"p_he": p, "evidence": "visible"}))["p_he"], p)

    def test_invalid_probability_and_schema_are_rejected(self):
        for p in (
            True,
            False,
            "0.8",
            "80%",
            80,
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            float("-inf"),
            [],
            {},
        ):
            with self.subTest(p=p), self.assertRaises(ValueError):
                parse(json.dumps({"p_he": p, "evidence": "visible"}))
        for text in (
            '{"label":1,"evidence":"x"}',
            '{"p_he":0.5}',
            '{"p_he":0.5,"evidence":""}',
            '{"p_he":0.5,"evidence":"x","label":1}',
            '{"p_he":0.1,"p_he":0.9,"evidence":"x"}',
            "[]",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_scores_against_hand_computed_values(self):
        m = a.metrics([row(0, 0.25), row(1, 0.75)], 2)
        self.assertAlmostEqual(m["brier_score_on_valid"], 0.0625)
        self.assertAlmostEqual(m["log_loss_on_valid"], -math.log(0.75))
        self.assertEqual(m["roc_auc_on_valid"], 1)
        self.assertEqual(a.metrics([row(0, 0.5), row(1, 0.5)], 2)["roc_auc_on_valid"], 0.5)
        self.assertEqual(a.metrics([row(0, 0.9), row(1, 0.1)], 2)["roc_auc_on_valid"], 0)

    def test_threshold_changes_only_auxiliary_classification(self):
        rows = [row(0, 0.5), row(1, 0.75)]
        m1 = a.metrics(rows, 2, 0.5)
        m2 = a.metrics(rows, 2, 0.8)
        self.assertEqual(m1["confusion_matrix"], {"tn": 0, "fp": 1, "fn": 0, "tp": 1})
        self.assertEqual(m2["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 1, "tp": 0})
        for key in ("brier_score_on_valid", "log_loss_on_valid", "roc_auc_on_valid"):
            self.assertEqual(m1[key], m2[key])

    def test_missing_predictions_single_class_and_extremes(self):
        rows = [
            {"status": "error", "p_he": None, "true_label": 0},
            {"status": "abstain", "p_he": None, "true_label": 1},
        ]
        for inputs in ([], rows):
            m = a.metrics(inputs, 3)
            for key in ("brier_score_on_valid", "log_loss_on_valid", "roc_auc_on_valid"):
                self.assertIsNone(m[key])
            self.assertEqual(m["coverage"], 0)
        m = a.metrics([row(1, 0)], 1)
        self.assertIsNone(m["roc_auc_on_valid"])
        self.assertEqual(m["brier_score_on_valid"], 1)
        self.assertTrue(math.isfinite(m["log_loss_on_valid"]))
        self.assertAlmostEqual(m["log_loss_on_valid"], -math.log(1e-15))
        self.assertTrue(math.isfinite(a.metrics([row(0, 1)], 1)["log_loss_on_valid"]))

    def test_invalid_threshold_rejected_before_reading_images(self):
        args = SimpleNamespace(limit=0, max_tokens=1, timeout=1, retries=0)
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True):
            args.threshold = threshold
            with self.subTest(threshold=threshold), patch.object(a, "load_samples") as read:
                with self.assertRaises(ValueError):
                    a.run_probability(args)
                read.assert_not_called()

    def test_reference_workflow_preserves_probability_and_null(self):
        import hashlib

        from PIL import Image
        from test_knowledge import FakeModel

        from models import ImagePart

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            rows = []
            for i in range(6):
                path = root / "images" / f"{i}.png"
                Image.new("L", (8, 8), color=i).save(path)
                rows.append(
                    {
                        "image": f"images/{i}.png",
                        "label": i % 2,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_dataset": "test",
                    }
                )
            (root / "labels/labels.txt").write_text(
                "".join(f"{r['image']} {r['label']}\n" for r in rows)
            )
            (root / "labels/manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            split = {
                "reference_pool": [r["image"] for r in rows[:4]],
                "references": [r["image"] for r in rows[:4]],
                "test": [r["image"] for r in rows[4:]],
                "samples": {r["image"]: {**r, "group_id": str(i)} for i, r in enumerate(rows)},
            }
            sp = root / "split.json"
            sp.write_text(json.dumps(split))
            args = SimpleNamespace(
                dataset=root,
                output=root / "run",
                mode="reference",
                split=sp,
                prompt=None,
                threshold=0.9,
                limit=0,
                seed=42,
                model=None,
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=False,
            )
            client = FakeModel()
            probabilities = iter([0.8123, None])

            def generate(request):
                client.requests.append(request)
                return ModelResponse(
                    text=json.dumps({"p_he": next(probabilities), "evidence": "observed"}),
                    completed=True,
                )

            client.generate = generate
            self.assertEqual(a.run_probability(args, client), 0)
            records = [
                json.loads(line)
                for line in (args.output / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["p_he"], 0.8123)
            self.assertAlmostEqual(records[0]["p_non_he"], 0.1877)
            self.assertEqual(records[0]["predicted_label"], 0)
            self.assertEqual(records[1]["status"], "abstain")
            self.assertIsNone(records[1]["p_non_he"])
            self.assertEqual(sum(isinstance(x, ImagePart) for x in client.requests[0].content), 5)


if __name__ == "__main__":
    unittest.main()
