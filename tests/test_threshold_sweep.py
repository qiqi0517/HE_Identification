"""离线阈值重放：同一初判、独立阈值、缓存缺失与真实失败隔离。"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
import sweep_thresholds as sweep


def prediction(p, status=None):
    return dict(
        status=status or ("abstain" if p is None else "ok"),
        p_he=p,
        usage={"total_tokens": 10},
        latency_seconds=1,
    )


class ThresholdSweepTests(unittest.TestCase):
    def fixture(self, root, review=True):
        source = root / "run"
        source.mkdir()
        prompt = b"original prompt"
        second = b"review prompt"
        config = dict(
            selected=6,
            decision_threshold=0.5,
            dry_run=False,
            prompt_sha256=hashlib.sha256(prompt).hexdigest(),
            review=dict(
                enabled=review,
                lower_threshold=0.1,
                upper_threshold=0.5,
                prompt_sha256=hashlib.sha256(second).hexdigest(),
            ),
        )
        rows = []
        samples = []
        for i, (p1, p2, label) in enumerate(
            [
                (0.2, 0.8, 1),
                (0.4, 0.1, 1),
                (0.45, 0.7, 0),
                (0.5, None, 1),
                (0.1, None, 0),
                (None, None, 1),
            ]
        ):
            sample = dict(image=f"images/{i}.png", label=label, source_dataset="test")
            samples.append(sample)
            first = prediction(p1)
            second_stage = prediction(p2) if review and p1 is not None and 0.1 < p1 < 0.5 else None
            row = dict(
                image=sample["image"],
                true_label=label,
                source_dataset="test",
                **(second_stage if second_stage is not None else first),
            )
            if review:
                row.update(
                    initial_prediction=first,
                    review_prediction=second_stage,
                    reviewed=second_stage is not None,
                )
            rows.append(row)
        for name, obj in [("run.json", config), ("samples.json", samples)]:
            (source / name).write_text(json.dumps(obj))
        (source / "prompt.txt").write_bytes(prompt)
        (source / "review_prompt.txt").write_bytes(second)
        self.write_rows(source, rows)
        return source, rows

    def write_rows(self, source, rows):
        (source / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def args(self, source, output, **override):
        values = dict(
            run=source,
            output=output,
            review_thresholds=[0.1, 0.2, 0.3],
            decision_thresholds=[0.3, 0.5],
            review_upper=None,
        )
        values.update(override)
        return SimpleNamespace(**values)

    def test_frozen_grid_and_independent_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = self.fixture(Path(d))
            _, samples, frozen, _ = sweep.load_run(source)
            results, common = sweep.sweep(frozen, samples, [0.1, 0.2], [0.3, 0.5], 0.5)
            self.assertEqual(len(results), 6)
            for index in (0, 2, 4):
                a, b = results[index : index + 2]
                self.assertEqual(a["requested_reviews"], b["requested_reviews"])
                self.assertEqual(
                    [r["p_he"] for r in a["predictions"]], [r["p_he"] for r in b["predictions"]]
                )
                self.assertEqual(
                    a["metrics"]["log_loss_on_valid"], b["metrics"]["log_loss_on_valid"]
                )
            low01 = results[2]
            self.assertEqual(low01["requested_reviews"], 3)
            self.assertEqual(low01["changes_on_paired_valid"]["introduced_fn"], 1)
            self.assertEqual(low01["changes_on_paired_valid"]["rescued_fn"], 1)
            self.assertEqual(results[4]["requested_reviews"], 2)
            self.assertEqual(len(common), 5)

    def test_missing_cache_skips_entire_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = self.fixture(Path(d))
            _, samples, frozen, _ = sweep.load_run(source)
            results, _ = sweep.sweep(frozen, samples, [0.05], [0.5], 0.5)
            self.assertEqual(results[0]["status"], "ok")
            self.assertEqual(results[1]["status"], "unavailable")
            self.assertIsNone(results[1]["metrics"])
            self.assertEqual(results[1]["missing_reviews"], ["images/4.png"])
            self.assertNotIn("predictions", results[1])

    def test_error_and_abstention_are_cached_not_missing(self):
        for status in ("error", "abstain"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                source, rows = self.fixture(Path(d))
                rows[0].update(prediction(None, status))
                rows[0]["review_prediction"] = prediction(None, status)
                self.write_rows(source, rows)
                _, samples, frozen, _ = sweep.load_run(source)
                results, common = sweep.sweep(frozen, samples, [0.1, 0.2], [0.5], 0.5)
                self.assertEqual(results[1]["status"], "ok")
                self.assertEqual(results[1]["metrics"]["valid_predictions"], 4)
                self.assertEqual(results[2]["metrics"]["valid_predictions"], 5)
                self.assertEqual(len(common), 4)
                self.assertTrue(all(r["common_valid_metrics"]["planned"] == 4 for r in results))

    def test_single_pass_input_still_supports_classification_sweep(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = self.fixture(Path(d), review=False)
            _, samples, frozen, _ = sweep.load_run(source)
            results, _ = sweep.sweep(frozen, samples, [0.1], [0.3, 0.5], 0.5)
            self.assertEqual(
                [r["status"] for r in results], ["ok", "ok", "unavailable", "unavailable"]
            )

    def test_output_snapshot_no_network_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, _ = self.fixture(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            args = self.args(source, root / "sweep")
            with (
                patch("he_agent.create_model", side_effect=AssertionError("no model")),
                patch("urllib.request.urlopen", side_effect=AssertionError("no network")),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(sweep.run(args), 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["network_calls"], 0)
            self.assertEqual(manifest["review_upper"], 0.5)
            self.assertEqual(
                (args.output / "source/predictions.jsonl").read_bytes(), before["predictions.jsonl"]
            )
            self.assertTrue((args.output / "case_000/predictions.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                sweep.run(args)
            with self.assertRaisesRegex(ValueError, "outside"):
                sweep.run(self.args(source, source / "sweep"))

    def test_hash_mismatch_rejected(self):
        for filename in ("prompt.txt", "review_prompt.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as d:
                source, _ = self.fixture(Path(d))
                (source / filename).write_text("changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    sweep.load_run(source)

    def test_incomplete_duplicate_order_and_truth_rejected(self):
        for change in ("partial", "duplicate", "reorder", "truth", "gate", "final"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                source, rows = self.fixture(Path(d))
                if change == "partial":
                    rows.pop()
                elif change == "duplicate":
                    rows[1] = rows[0]
                elif change == "reorder":
                    rows.reverse()
                elif change == "truth":
                    rows[0]["true_label"] = 0
                elif change == "gate":
                    rows[0]["reviewed"] = False
                elif change == "final":
                    rows[0]["p_he"] = 0.99
                self.write_rows(source, rows)
                with self.assertRaises(ValueError):
                    sweep.load_run(source)

    def test_invalid_probabilities_and_status_rejected(self):
        for p in (True, float("nan"), float("inf"), -1, 1.1):
            with self.subTest(p=p), self.assertRaises(ValueError):
                sweep.validate_prediction(dict(status="ok", p_he=p))
        with self.assertRaises(ValueError):
            sweep.validate_prediction(dict(status="error", p_he=0.2))

    def test_invalid_thresholds_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, _ = self.fixture(root)
            for override in [
                dict(review_thresholds=[0.5]),
                dict(review_thresholds=[float("nan")]),
                dict(decision_thresholds=[float("inf")]),
                dict(review_upper=0),
            ]:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    sweep.run(self.args(source, root / "out", **override))
                self.assertFalse((root / "out").exists())

    def test_dry_run_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = self.fixture(Path(d))
            config = json.loads((source / "run.json").read_text())
            config["dry_run"] = True
            (source / "run.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "dry-run"):
                sweep.load_run(source)


if __name__ == "__main__":
    unittest.main()
