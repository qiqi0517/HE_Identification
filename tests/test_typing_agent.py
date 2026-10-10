"""Typing output, confidence review, reference isolation and all four modes."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import he_agent
import typing_agent as agent
from models import ImagePart, ModelError, ModelResponse, TextPart
from reference_split import validate_typing_split
from selective_review import needs_confidence_review
from typing_protocol import CLASSES, parse_prediction


def response(label="CE2", confidence=0.9, status="ok"):
    return ModelResponse(
        text=json.dumps(
            dict(
                status=status,
                predicted_type=label,
                confidence=confidence,
                findings=["visible structure"],
            )
        ),
        completed=True,
    )


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.requests = []
        self.credentials = 0

    def describe(self):
        return {"provider": "fake", "model": "typing-test"}

    def validate_credentials(self):
        self.credentials += 1

    def generate(self, request):
        self.requests.append(request)
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TypingAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        (self.data / "labels").mkdir()
        partitions = {r: [] for r in ("reference", "search", "selection", "test")}
        rows, saved = [], {}
        # Two search samples allow fixed-denominator assertions after a fatal stop.
        specs = [("reference", label) for label in CLASSES] + [
            ("search", "CE2"),
            ("search", "NO_HE"),
            ("selection", "AE"),
            ("test", "CE1"),
        ]
        for i, (role, label) in enumerate(specs):
            name = f"images/sample{i}.png"
            Image.new("RGB", (12, 11), (i * 15, 20, 30)).save(self.data / name)
            digest = hashlib.sha256((self.data / name).read_bytes()).hexdigest()
            rows.append(
                dict(
                    image=name,
                    typing_label=label,
                    primary_label_eligible=True,
                    sha256=digest,
                    sources=["fixture-source-secret"],
                )
            )
            saved[name] = dict(label=label, sha256=digest, group_id=f"g{i}")
            partitions[role].append(name)
        (self.data / "labels/manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (self.data / "labels/labels_typing.txt").write_text(
            "".join(f"{r['image']} {r['typing_label']}\n" for r in rows)
        )
        self.split = self.root / "split.json"
        self.split.write_text(
            json.dumps(dict(schema="he-typing-split-v1", partitions=partitions, samples=saved))
        )

    def args(self, **overrides):
        values = dict(
            dataset=self.data,
            split=self.split,
            role="search",
            mode="baseline",
            reference_count=None,
            limit=1,
            seed=0,
            max_tokens=1000,
            timeout=30,
            retries=0,
            review=False,
            review_threshold=0.8,
            prompt=None,
            review_prompt=None,
            provider="fake",
            model="fake",
            output=self.root / "run",
            dry_run=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def run_agent(self, args, client):
        with contextlib.redirect_stdout(io.StringIO()):
            return he_agent.run(args, client)

    def test_parser_rejects_invalid_and_truncated(self):
        for c in (-0.1, 1.1, True, "0.5", None, float("nan"), float("inf")):
            with self.subTest(confidence=c), self.assertRaises(ValueError):
                parse_prediction(response(confidence=c))
        for answer in (
            response(label="CE6"),
            response(None, None),
            ModelResponse(text=response().text, completed=False),
            ModelResponse(
                text=response().text.replace(
                    '"confidence": 0.9', '"confidence": 0.9, "confidence": 0.1'
                ),
                completed=True,
            ),
        ):
            with self.assertRaises(ValueError):
                parse_prediction(answer)
        self.assertEqual(parse_prediction(response(confidence=0))["confidence"], 0)
        self.assertEqual(parse_prediction(response(confidence=1))["confidence"], 1)
        self.assertIsNone(parse_prediction(response(None, None, "abstain"))["predicted_type"])

    def test_cl_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown type"):
            parse_prediction(response("CL"))

    def test_review_boundary_all_labels_and_invalid(self):
        for label in CLASSES:
            self.assertTrue(
                needs_confidence_review(
                    dict(status="ok", predicted_type=label, confidence=0.79), 0.8
                )
            )
        self.assertFalse(needs_confidence_review(dict(status="ok", confidence=0.8), 0.8))
        self.assertFalse(needs_confidence_review(dict(status="error", confidence=0.1), 0.8))
        self.assertFalse(needs_confidence_review(dict(status="ok", confidence=True), 0.8))
        self.assertFalse(needs_confidence_review(dict(status="ok", confidence=0), 0))

    def test_four_modes_reference_order_knowledge_and_no_query_metadata(self):
        for mode in ("baseline", "reference", "knowledge", "reference_knowledge"):
            with self.subTest(mode=mode):
                client = FakeClient([response()])
                args = self.args(mode=mode, output=self.root / mode)
                self.assertEqual(self.run_agent(args, client), 0)
                req = client.requests[0]
                has_refs = mode in ("reference", "reference_knowledge")
                self.assertEqual(
                    sum(isinstance(p, ImagePart) for p in req.content), 8 if has_refs else 1
                )
                self.assertEqual(
                    "分型知识（依据" in req.system_prompt,
                    mode in ("knowledge", "reference_knowledge"),
                )
                texts = "\n".join(p.text for p in req.content if isinstance(p, TextPart))
                self.assertNotIn("fixture-source-secret", texts)
                self.assertNotIn("images/", texts)
                if has_refs:
                    ref_texts = [
                        p.text
                        for p in req.content
                        if isinstance(p, TextPart) and p.text.startswith("参考图 ")
                    ]
                    for label, text in zip(CLASSES, ref_texts):
                        self.assertIn(f"分型标签：{label}", text)
                run = json.loads((args.output / "run.json").read_text())
                self.assertEqual(run["output_schema"], "he-typing-confidence-v1")
                self.assertNotIn("decision_threshold", run)

    def test_low_confidence_independent_review_replaces_label_and_confidence(self):
        client = FakeClient([response("NO_HE", 0.2), response("CE2", 0.95)])
        args = self.args(review=True, mode="reference_knowledge")
        self.assertEqual(self.run_agent(args, client), 0)
        rows = [
            json.loads(line)
            for line in (args.output / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertEqual(rows[0]["predicted_type"], "CE2")
        self.assertEqual(rows[0]["confidence"], 0.95)
        self.assertEqual(rows[0]["initial_prediction"]["confidence"], 0.2)
        self.assertEqual(client.requests[0].content, client.requests[1].content)
        self.assertNotIn('"confidence": 0.2', client.requests[1].system_prompt)
        self.assertEqual(len((args.output / "stages.jsonl").read_text().splitlines()), 2)

    def test_equal_confidence_does_not_review(self):
        client = FakeClient([response(confidence=0.8)])
        self.run_agent(self.args(review=True), client)
        self.assertEqual(len(client.requests), 1)

    def test_review_checklist_reference_toggle_and_knowledge_isolation(self):
        for policy, image_count in (("none", 1), ("same", 8)):
            with self.subTest(reference_policy=policy):
                client = FakeClient([response("NO_HE", 0.2), response("CE2", 0.7)])
                args = self.args(
                    review=True,
                    mode="reference_knowledge",
                    review_reference=policy,
                    review_knowledge="checklist",
                    output=self.root / ("checklist_" + policy),
                )
                self.assertEqual(self.run_agent(args, client), 0)
                initial, review = client.requests
                self.assertEqual(sum(isinstance(p, ImagePart) for p in initial.content), 8)
                self.assertEqual(sum(isinstance(p, ImagePart) for p in review.content), image_count)
                self.assertIn("复核检查表", review.system_prompt)
                self.assertIn("分型知识（", initial.system_prompt)
                self.assertNotIn("分型知识（", review.system_prompt)
                self.assertNotIn('"confidence": 0.2', review.system_prompt)
                config = json.loads((args.output / "run.json").read_text())
                self.assertEqual(config["review"]["reference_count"], image_count - 1)
                self.assertEqual(config["review"]["knowledge"], "checklist")
                self.assertTrue((args.output / "review_knowledge_sources.json").is_file())

    def test_review_knowledge_none_and_same(self):
        for choice in ("none", "same"):
            client = FakeClient([response(confidence=0.1), response()])
            args = self.args(
                review=True,
                mode="knowledge",
                review_knowledge=choice,
                output=self.root / ("knowledge_" + choice),
            )
            self.run_agent(args, client)
            prompt = client.requests[1].system_prompt
            self.assertNotIn("复核检查表", prompt)
            self.assertEqual("分型知识（" in prompt, choice == "same")

    def test_review_failure_in_fixed_denominator_and_fatal_stops(self):
        client = FakeClient([response(confidence=0.1), ModelError("fake fatal", fatal=True)])
        args = self.args(review=True, limit=0)
        self.assertEqual(self.run_agent(args, client), 1)
        saved = json.loads((args.output / "metrics.json").read_text())
        self.assertEqual(saved["planned"], 2)
        self.assertEqual(saved["errors"], 1)
        self.assertEqual(saved["unattempted"], 1)
        self.assertEqual(saved["accuracy"], 0)
        final = json.loads((args.output / "predictions.jsonl").read_text().splitlines()[0])
        self.assertIsNone(final["confidence"])
        self.assertIsNone(final["predicted_type"])

    def test_abstain_does_not_trigger_review(self):
        client = FakeClient([response(None, None, "abstain")])
        args = self.args(review=True)
        self.assertEqual(self.run_agent(args, client), 0)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(json.loads((args.output / "metrics.json").read_text())["accuracy"], 0)

    def test_dry_run_does_not_read_credentials(self):
        client = FakeClient([])
        self.assertEqual(self.run_agent(self.args(dry_run=True, review=True), client), 0)
        self.assertEqual(client.credentials, 0)
        self.assertEqual(client.requests, [])

    def test_split_stale_labels_and_group_leakage_rejected(self):
        samples = agent.load_samples(self.data)
        split = json.loads(self.split.read_text())
        validate_typing_split(split, samples)
        query = split["partitions"]["search"][0]
        ref = split["partitions"]["reference"][0]
        split["samples"][query]["group_id"] = split["samples"][ref]["group_id"]
        with self.assertRaises(ValueError):
            validate_typing_split(split, samples)
        split = json.loads(self.split.read_text())
        split["samples"][query]["label"] = "CE5"
        with self.assertRaises(ValueError):
            validate_typing_split(split, samples)
        split["schema"] = "he-auto-harness-split-v1"
        with self.assertRaises(ValueError):
            validate_typing_split(split, samples)


if __name__ == "__main__":
    unittest.main()
