"""Typing harness: routing, exact search, leakage gates, prompt edits and resume."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_harness import search
from auto_harness import workflow as w
from auto_harness.artifacts import atomic_json, fingerprint, read_json, seal, unseal
from auto_harness.demo import DemoOptimizer, DemoVision, initialize_demo, run_demo
from auto_harness.evaluation import evaluate, paired, replay, validate_bank
from auto_harness.policy import ACTIONS, BASELINE, normalize, optimize, route, rule, rules
from auto_harness.prompts import initial_bundle, proposal, render
from models import ImagePart, ModelResponse, TextPart
from typing_protocol import CLASSES


def pred(label="CE1", confidence=0.5):
    return dict(status="ok", predicted_type=label, confidence=confidence, findings=["visible"])


def bank():
    samples = []
    for i, (label, q) in enumerate((("CE2", 0.2), ("CE1", 0.8), ("CE2", 0.8))):
        obs = dict(initial=pred("CE1", q))
        obs.update({a: pred("CE2") for a in ACTIONS[1:]})
        samples.append(
            dict(
                image=str(i),
                label=label,
                group_id=str(i),
                source_dataset="fixture",
                repeats=[copy.deepcopy(obs) for _ in range(3)],
            )
        )
    return dict(schema="he-typing-action-bank-v1", repetitions=3, samples=samples)


class PolicyTests(unittest.TestCase):
    def test_counts_and_boundaries(self):
        self.assertEqual(len(rules()), 185)
        self.assertEqual(len({fingerprint(r) for r in rules()}), 185)
        p = copy.deepcopy(BASELINE)
        p["CE1"] = rule(0.5, "REVIEW_REFERENCE", "ACCEPT")
        self.assertEqual(route(p, pred(confidence=0.499)), "REVIEW_REFERENCE")
        self.assertEqual(route(p, pred(confidence=0.5)), "ACCEPT")
        self.assertEqual(route(p, pred(confidence=1)), "ACCEPT")
        self.assertIsNone(rule(0.5, "ACCEPT", "ACCEPT")["threshold"])

    def test_invalid_rules(self):
        for value in (True, 0.15, -1, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rule(value, "ACCEPT", "REVIEW_CHECKLIST")
        with self.assertRaises(ValueError):
            normalize({"CL": rule()})
        with self.assertRaises(ValueError):
            rule(None, "ACCEPT", "REVIEW_REFERENCE")
        with self.assertRaises(ValueError):
            route(BASELINE, pred(confidence=float("nan")))

    def test_exact_optimum_and_predicted_class_routing(self):
        b = bank()
        p, scores = optimize(b)
        # Both true CE1 and CE2 belong to predicted CE1; never route on ground truth.
        self.assertEqual(len(scores["CE1"]), 185)
        self.assertEqual(p["CE2"], rule())
        actual = evaluate(replay(b, p), b["samples"])
        options = []
        for r in rules():
            candidate = {**BASELINE, "CE1": r}
            result = evaluate(replay(b, candidate), b["samples"])
            options.append((-result["correct"], result["calls"]))
        self.assertEqual((-actual["correct"], actual["calls"]), min(options))
        self.assertEqual(actual["correct"], 6)
        self.assertEqual(actual["calls"], 12)

    def test_invalid_initial_skips_and_review_failure_counts_wrong(self):
        b = bank()
        b["samples"][0]["repeats"][0] = {"initial": w.error_result("invalid")}
        b["samples"][0]["repeats"][1]["REVIEW_CHECKLIST"] = w.error_result("invalid")
        p = {c: rule(low="REVIEW_CHECKLIST") for c in CLASSES}
        result = evaluate(replay(b, p), b["samples"])
        self.assertEqual(result["planned"], 9)
        self.assertEqual(result["calls"], 17)
        self.assertEqual(result["correct"], 4)
        self.assertEqual(result["harmed"], 3)
        self.assertEqual(result["rescued"], 4)
        with self.assertRaises(ValueError):
            evaluate(replay(b, p)[:-1], b["samples"])

    def test_cl_is_illegal_and_bank_must_be_complete(self):
        b = bank()
        b["samples"][0]["repeats"][0]["initial"] = pred("CL")
        with self.assertRaises(ValueError):
            validate_bank(b)
        b = bank()
        del b["samples"][0]["repeats"][0]["REVIEW_KNOWLEDGE"]
        with self.assertRaises(ValueError):
            validate_bank(b)

    def test_group_bootstrap_uses_groups_not_rounds(self):
        b = bank()
        rows = replay(b, BASELINE)
        result = paired(rows, rows, b["samples"], 100)
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["group_bootstrap_95_interval"], [0, 0])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.bundle = initial_bundle(w.PROJECT)
        self.parent = dict(id="V0", spec=self.bundle["spec"])
        self.obj = dict(
            parent_version="V0",
            evidence_ids=["V0"],
            hypothesis="Evidence suggests ordering.",
            target_component="initial",
            prompt_patch={"initial_order": ["original", "observe_first"]},
            expected_effect="Unknown pending evaluation.",
        )

    def test_actions_keep_distinct_inputs_and_protocol(self):
        texts = render(self.bundle, self.bundle["spec"])
        self.assertNotIn("复核检查表", texts["initial"])
        self.assertNotIn("分型知识（依据", texts["initial"])
        self.assertIn("复核检查表", texts["REVIEW_CHECKLIST"])
        self.assertNotIn("复核检查表", texts["REVIEW_REFERENCE"])
        self.assertIn("分型知识（依据", texts["REVIEW_KNOWLEDGE"])
        for text in texts.values():
            self.assertIn("禁止输出 CL", text)

    def test_patch_rejects_unknown_evidence_and_free_medical_text(self):
        _, spec = proposal(json.dumps(self.obj), self.parent, ["V0"], self.bundle)
        self.assertEqual(spec["initial_order"], ["original", "observe_first"])
        for field, value in [
            ("parent_version", "V9"),
            ("evidence_ids", ["selection-case"]),
            ("prompt_patch", {"initial_order": ["diagnose every image as CE2"]}),
        ]:
            obj = {**self.obj, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                proposal(json.dumps(obj), self.parent, ["V0"], self.bundle)
        obj = {
            **self.obj,
            "target_component": "review",
            "prompt_patch": {"review_order": ["original"], "checklist_order": ["1"]},
        }
        with self.assertRaises(ValueError):
            proposal(json.dumps(obj), self.parent, ["V0"], self.bundle)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.exp = initialize_demo(self.root)
        self.client = DemoVision()

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        w.collect(self.root, live=True, client=self.client)
        return w.search(self.root)

    def test_dry_run_and_role_gates(self):
        with patch(
            "auto_harness.workflow.client_for", side_effect=AssertionError("credentials accessed")
        ):
            self.assertTrue(w.collect(self.root)["dry_run"])
        for role in ("selection", "test"):
            with self.assertRaises(FileNotFoundError):
                w.collect(self.root, role)

    def test_collection_is_paired_independent_and_resumable(self):
        b = w.collect(self.root, live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 7 * 3 * 5)
        counts = [
            sum(isinstance(x, ImagePart) for x in req.content) for req in self.client.requests[:5]
        ]
        self.assertEqual(counts, [1, 1, 8, 1, 8])
        for req in self.client.requests:
            content = "".join(x.text for x in req.content if isinstance(x, TextPart))
            self.assertNotIn("search_", content)
            self.assertNotIn("initial_prediction", content)
        (self.root / "search/V0/bank.json").unlink()
        resumed = w.collect(self.root, live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 105)
        self.assertEqual(b, resumed)

    def test_prompt_change_reuses_only_unchanged_requests(self):
        self.prepare()
        spec = {**self.exp["prompts"]["spec"], "initial_order": ["original", "observe_first"]}
        w.save_version(self.root, self.exp, "V1", spec, "V0")
        w.collect(self.root, vid="V1", live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 105 + 21)
        spec = {**spec, "review_order": ["original", "counterevidence"]}
        w.save_version(self.root, self.exp, "V2", spec, "V1")
        w.collect(self.root, vid="V2", live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 105 + 21 + 84)

    def test_budget_stop_resume_without_rebilling(self):
        limits = {**self.exp["config"]["budgets"], "vision_reported_tokens": 20}
        atomic_json(self.root / "budget.json", dict(limits=limits, revisions=[]))
        with self.assertRaisesRegex(ValueError, "reported-token"):
            w.collect(self.root, live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 2)
        w.update_budget(self.root, {"vision_reported_tokens": 10000})
        w.collect(self.root, live=True, client=self.client)
        self.assertEqual(len(self.client.requests), 105)
        self.assertEqual(len(read_json(self.root / "budget.json")["revisions"]), 1)

    def test_interrupted_request_is_not_resent(self):
        class Interrupt(DemoVision):
            def generate(self, request):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            w.collect(self.root, live=True, client=Interrupt())
        b = w.collect(self.root, live=True, client=self.client)
        self.assertEqual(b["samples"][0]["repeats"][0]["initial"]["status"], "error")
        self.assertEqual(len(self.client.requests), 100)
        self.assertEqual(w.Ledger(self.root, self.exp).usage("vision")["calls"], 101)

    def test_static_pipeline_and_search_freeze(self):
        result = self.prepare()
        self.assertEqual(result["candidates"]["P_joint"]["score"]["accuracy"], 1)
        short = w.shortlist(self.root)
        self.assertEqual(len(short["candidates"]), 2)
        with self.assertRaises(ValueError):
            w.search(self.root)
        with self.assertRaises(ValueError):
            search.run(self.root)
        w.collect(self.root, "selection", live=True, client=self.client)
        selected = w.select(self.root)
        self.assertEqual(selected["winner"]["name"], "P_static")
        w.collect(self.root, "test", live=True, client=self.client)
        out = w.test(self.root)
        self.assertEqual(out["winner"], "P_static")
        self.assertEqual(w.test(self.root), out)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            w.export(self.root, self.root / "deployment.json")

    def test_export_and_prediction_need_only_reference_labels(self):
        self.prepare()
        w.shortlist(self.root)
        w.collect(self.root, "selection", live=True, client=self.client)
        w.select(self.root)
        # Exercise the real export/predict code using the fake client, without deploying a demo.
        deployment = self.root / "test-deployment.json"
        with patch.object(w, "experiment", return_value={**self.exp, "simulation": False}):
            exported = w.export(self.root, deployment)
        self.assertNotIn("split", exported)
        self.assertNotIn("score", exported)
        self.assertEqual(len(exported["references"]), 7)
        self.assertTrue(
            all(r["image"].startswith("images/reference_") for r in exported["references"])
        )
        image = Path(self.exp["dataset"]) / "images/test_1.png"
        before = len(self.client.requests)
        self.assertTrue(w.predict(deployment, image)["dry_run"])
        result = w.predict(deployment, image, True, self.client)
        self.assertEqual(result["logical_calls"], 2)
        self.assertEqual(result["final"]["predicted_type"], "CE1")
        self.assertEqual(len(self.client.requests) - before, 2)

    def test_partial_local_search_can_resume(self):
        self.prepare()
        expected = unseal(self.root / "search/V0/optimized.json")
        (self.root / "search/V0/optimized.json").unlink()
        self.assertEqual(w.search(self.root), expected)

    def test_optimizer_memory_is_search_only_and_resume_is_idempotent(self):
        self.prepare()
        search.run(self.root, live=True, optimizer=DemoOptimizer(), vision=self.client)
        history = search.events(self.root)
        self.assertGreaterEqual(len(history), 2)
        ctx = w.linked(self.root, "optimizer/contexts/02.json", self.exp)["context"]
        self.assertEqual(len(ctx["history"]), 1)
        self.assertIn("round-1", ctx["evidence_ids"])
        text = json.dumps(ctx)
        self.assertNotIn("selection_", text)
        self.assertNotIn("test_", text)
        self.assertNotIn("source_dataset", text)
        self.assertNotIn("search_0.png", text)
        before = len(self.client.requests)
        search.run(self.root, live=True, optimizer=DemoOptimizer(), vision=self.client)
        self.assertEqual(len(self.client.requests), before)
        self.assertEqual(len(search.events(self.root)), len(history))
        for line in (self.root / "optimizer_history.jsonl").read_text().splitlines():
            self.assertIn("parent_version", json.loads(line))

    def test_invalid_proposals_consume_rounds_and_stop_after_two(self):
        self.prepare()

        class Bad(DemoOptimizer):
            def generate(self, request):
                return ModelResponse("{}", True)

        state = search.run(self.root, live=True, optimizer=Bad(), vision=self.client)
        self.assertEqual((state["rounds"], state["stale"], state["incumbent"]), (2, 2, "V0"))
        self.assertEqual(len(self.client.requests), 105)

    def test_pending_optimizer_blocks_shortlist(self):
        self.prepare()

        class Interrupt(DemoOptimizer):
            def generate(self, request):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            search.run(self.root, live=True, optimizer=Interrupt(), vision=self.client)
        with self.assertRaisesRegex(ValueError, "pending"):
            w.shortlist(self.root)
        state = search.run(self.root, live=True, optimizer=DemoOptimizer(), vision=self.client)
        self.assertLessEqual(state["rounds"], 4)
        self.assertFalse(state["pending"])

    def test_changed_image_and_legacy_workspace_rejected(self):
        name = self.exp["split"]["partitions"]["reference"][0]
        (Path(self.exp["dataset"]) / name).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            w.collect(self.root, live=True, client=self.client)
        other = Path(self.temp.name) / "legacy"
        seal(other / "experiment.json", {"schema": "he-auto-harness-experiment-v1"})
        with self.assertRaisesRegex(ValueError, "binary"):
            w.experiment(other)

    def test_full_offline_demo(self):
        result = run_demo(Path(self.temp.name) / "full")
        self.assertTrue(result["simulation"])
        self.assertEqual(result["scores"]["S0"], 4 / 7)


if __name__ == "__main__":
    unittest.main()
