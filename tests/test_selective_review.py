"""验证选择性复核分流边界、替换策略、配对评分与请求隔离。"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
import test_experiment_matrix as fixtures
from test_he_agent import AlternateClient, response

import he_agent as agent
from models import ModelError, ModelResponse, create_model
from prompt_profiles import MODES, load_prompt
from selective_review import needs_review, token_usage, validate_review_threshold


class SelectiveReviewTests(unittest.TestCase):
    def args(self, root, split, **overrides):
        values = dict(
            dataset=root,
            split=split,
            mode="reference",
            reference_count=8,
            prompt=None,
            provider="glm",
            model=None,
            output=root / "run",
            limit=0,
            seed=42,
            max_tokens=8192,
            timeout=1,
            retries=0,
            dry_run=False,
            threshold=0.5,
            review=True,
            review_threshold=0.2,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def execute(self, outputs, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = fixtures.MatrixTests().fixture(root, pool_size=8)
            args = self.args(root, split, **overrides)
            client = AlternateClient()
            capture = io.StringIO()
            with patch.object(client, "generate", side_effect=outputs) as generate:
                with redirect_stdout(capture):
                    code = agent.run(args, client)
            records = [
                json.loads(line)
                for line in (args.output / "predictions.jsonl").read_text().splitlines()
            ]
            stage_path = args.output / "stages.jsonl"
            stages = (
                [json.loads(line) for line in stage_path.read_text().splitlines()]
                if stage_path.exists()
                else []
            )
            return dict(
                code=code,
                rows=records,
                stages=stages,
                metrics=json.loads((args.output / "metrics.json").read_text()),
                config=json.loads((args.output / "run.json").read_text()),
                calls=[call.args[0] for call in generate.call_args_list],
                console=capture.getvalue(),
            )

    def test_strict_boundaries_and_abstentions(self):
        for p, expected in [
            (0, False),
            (0.2, False),
            (0.20001, True),
            (0.49999, True),
            (0.5, False),
            (1, False),
        ]:
            with self.subTest(p=p):
                self.assertEqual(needs_review(dict(status="ok", p_he=p), 0.2, 0.5), expected)
        for status in ("abstain", "error"):
            self.assertFalse(needs_review(dict(status=status, p_he=None), 0.2, 0.5))
        self.assertTrue(needs_review(dict(status="ok", p_he=0.6), 0.2, 0.7))

    def test_threshold_validation(self):
        for lower in (-0.1, 0.5, 0.6, float("nan"), float("inf"), True):
            with self.subTest(lower=lower), self.assertRaises(ValueError):
                validate_review_threshold(lower, 0.5)
        validate_review_threshold(0, 0.5)
        result = self.execute([response(0.2), response(0.5)])
        self.assertEqual(len(result["calls"]), 2)
        self.assertTrue(all(not row["reviewed"] for row in result["rows"]))
        result = self.execute([response(0.4), response(0.5)], review_threshold=0.4)
        self.assertEqual(len(result["calls"]), 2)

    def test_initial_flow_unchanged_when_disabled(self):
        result = self.execute([response(0.3), response(0.4)], review=False)
        self.assertEqual(len(result["calls"]), 2)
        self.assertFalse(result["stages"])
        self.assertNotIn("initial_prediction", result["rows"][0])
        self.assertNotIn("selective_review", result["metrics"])
        self.assertEqual(result["calls"][0].system_prompt, load_prompt("reference")[0])

    def test_paired_changes_and_no_answer_or_truth_leak(self):
        first = ModelResponse(
            text='{"p_he":0.314159,"evidence":"FIRST_ONLY_SENTINEL"}',
            completed=True,
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
        result = self.execute([first, response(0.8), first, response(0.9)])
        self.assertEqual(len(result["calls"]), 4)
        for index in (0, 2):
            initial, final = result["calls"][index : index + 2]
            self.assertEqual(initial.content, final.content)
            self.assertNotEqual(initial.system_prompt, final.system_prompt)
            payload_text = final.system_prompt + repr(final.content)
            for sentinel in (
                "FIRST_ONLY_SENTINEL",
                "0.314159",
                "private_case_",
                "private_source_marker",
                "true_label",
            ):
                self.assertNotIn(sentinel, payload_text)
        changes = result["metrics"]["selective_review"]["changes_on_paired_reviewed"]
        self.assertEqual(changes["rescued_fn"], 1)
        self.assertEqual(changes["introduced_fp"], 1)
        self.assertEqual(changes["probability_up"], 2)
        for row in result["rows"]:
            self.assertEqual(row["initial_prediction"]["p_he"], 0.314159)
            self.assertEqual(row["final_stage"], "review")
            self.assertEqual(row["logical_calls"], 2)
        self.assertEqual(len(result["stages"]), 4)
        self.assertIn("rescued_fn=1, introduced_fp=1", result["console"])
        self.assertEqual(result["config"]["review"]["lower_threshold"], 0.2)
        self.assertEqual(
            result["config"]["review"]["prompt_sha256"],
            hashlib.sha256(result["calls"][1].system_prompt.encode()).hexdigest(),
        )

    def test_review_can_lower_probability(self):
        result = self.execute([response(0.4), response(0.1), response(0.8)])
        self.assertEqual(result["rows"][0]["p_he"], 0.1)
        self.assertEqual(
            result["metrics"]["selective_review"]["changes_on_paired_reviewed"]["probability_down"],
            1,
        )
        self.assertAlmostEqual(
            result["metrics"]["brier_score_on_valid"],
            sum((r["p_he"] - r["true_label"]) ** 2 for r in result["rows"]) / 2,
        )

    def test_review_errors_null_and_truncation_do_not_fall_back(self):
        for failed, expected in [
            (response(None), "abstain"),
            (ModelError("temporary"), "error"),
            (response(0.9, completed=False), "error"),
            (ModelResponse(text="invalid", completed=True), "error"),
        ]:
            with self.subTest(expected=expected, failure=type(failed).__name__):
                result = self.execute([response(0.4), failed, response(0.8)])
                row = result["rows"][0]
                self.assertEqual(row["status"], expected)
                self.assertIsNone(row["p_he"])
                self.assertIsNone(row["predicted_label"])
                self.assertEqual(row["initial_prediction"]["p_he"], 0.4)
                self.assertEqual(result["metrics"]["selective_review"]["initial"]["coverage"], 1)
                self.assertEqual(result["metrics"]["coverage"], 0.5)
                self.assertEqual(result["metrics"]["selective_review"]["paired_valid"]["count"], 1)
                self.assertEqual(result["code"], int(expected == "error"))
                if isinstance(failed, ModelError):
                    self.assertIsNone(row["evidence"])
                    self.assertIsNone(row["raw_answer"])

    def test_initial_failure_and_null_skip_review(self):
        result = self.execute([response(None), ModelError("temporary")])
        self.assertEqual(len(result["calls"]), 2)
        self.assertEqual(result["metrics"]["selective_review"]["reviewed"], 0)

    def test_fatal_review_stops_batch(self):
        result = self.execute([response(0.4), ModelError("authorization failed", fatal=True)])
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["metrics"]["unattempted"], 1)
        self.assertEqual(len(result["stages"]), 2)

    def test_initial_is_persisted_before_interrupted_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = fixtures.MatrixTests().fixture(root, pool_size=8)
            args = self.args(root, split)
            with patch.object(
                AlternateClient, "generate", side_effect=[response(0.4), KeyboardInterrupt]
            ):
                with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                    agent.run(args, AlternateClient())
            lines = (args.output / "stages.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["p_he"], 0.4)

    def test_all_real_adapters_all_modes_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = fixtures.MatrixTests().fixture(root, pool_size=8)
            sample_lists = []
            for provider in ("glm", "openai", "deepseek"):
                for mode in MODES:
                    for dry_run in (False, True):
                        with self.subTest(provider=provider, mode=mode, dry_run=dry_run):
                            args = self.args(
                                root,
                                split,
                                provider=provider,
                                mode=mode,
                                reference_count=8 if "reference" in mode else None,
                                output=root / f"{provider}_{mode}_{dry_run}",
                                dry_run=dry_run,
                            )
                            client = create_model(provider)

                            def body(p):
                                answer = json.dumps(dict(p_he=p, evidence="visible"))
                                if provider != "openai":
                                    return {
                                        "choices": [
                                            {
                                                "message": {"content": answer},
                                                "finish_reason": "stop",
                                            }
                                        ]
                                    }
                                return {
                                    "status": "completed",
                                    "output": [
                                        {
                                            "type": "message",
                                            "role": "assistant",
                                            "status": "completed",
                                            "content": [{"type": "output_text", "text": answer}],
                                        }
                                    ],
                                }

                            with (
                                patch.object(client, "validate_credentials") as credentials,
                                patch.object(
                                    client, "_post", side_effect=[body(0.3), body(0.7), body(0.1)]
                                ) as post,
                                redirect_stdout(io.StringIO()),
                            ):
                                self.assertEqual(agent.run(args, client), 0)
                            self.assertEqual(post.call_count, 0 if dry_run else 3)
                            self.assertEqual(credentials.call_count, 0 if dry_run else 1)
                            sample_lists.append(
                                json.loads((args.output / "samples.json").read_text())
                            )
                            if dry_run:
                                self.assertFalse((args.output / "metrics.json").exists())
                                self.assertFalse((args.output / "stages.jsonl").exists())
                            else:

                                def content(payload):
                                    return (
                                        payload["input"]
                                        if provider == "openai"
                                        else payload["messages"][1:]
                                    )

                                self.assertEqual(
                                    content(post.call_args_list[0].args[0]),
                                    content(post.call_args_list[1].args[0]),
                                )
            self.assertTrue(all(samples == sample_lists[0] for samples in sample_lists))

    def test_reference_only_comparison_rejects_review(self):
        from compare_reference_runs import read_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"review": {"enabled": True}}))
            with self.assertRaisesRegex(ValueError, "selective-review"):
                read_run(root)

    def test_usage_missing_is_not_zero(self):
        result = token_usage(
            [
                {"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
                {"usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}},
                {},
            ]
        )
        self.assertEqual(result["total_tokens"], 35)
        self.assertEqual(result["logical_calls"], 3)
        self.assertEqual(result["stages_reporting"]["total_tokens"], 2)
        self.assertIsNone(token_usage([{}])["total_tokens"])


if __name__ == "__main__":
    unittest.main()
