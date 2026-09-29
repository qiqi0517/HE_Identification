"""不联网的协议、检索隔离、调用预算及端到端回归测试。"""

import copy
import io
import json
import os
import random
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from PIL import Image

from he_agent_v2 import LAB, PROJECT
from he_agent_v2.cli import parser, run
from he_agent_v2.evaluate import evaluate
from he_agent_v2.fixed_references import (
    FixedReferences,
    UnsafeReferences,
    fingerprint,
    sha,
)
from he_agent_v2.harness import infer_one
from he_agent_v2.knowledge_retrieval import KnowledgeLibrary
from he_agent_v2.routing import route
from he_agent_v2.schemas import Query, parse_observation
from he_agent_v2.transport import MeteredClient
from models import (
    ImagePart,
    ModelError,
    ModelRequest,
    ModelResponse,
    TextPart,
    create_model,
)
from reference_split import select_references


def observation(probability=0.1, tags=None):
    return {
        "quality": "good",
        "appearance_tags": tags or ["cystic"],
        "observations": ["可见液性结构"],
        "limitations": [],
        "p_he": probability,
        "evidence": "STAGE1_EVIDENCE_SENTINEL",
    }


def response(value, completed=True):
    return ModelResponse(json.dumps(value), completed, usage={"total_tokens": 10})


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value

    def describe(self):
        return {"provider": "mock", "model": "offline"}

    def validate_credentials(self):
        pass


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        self.samples = []
        for i in range(10):
            rng = random.Random(i)
            path = self.root / f"images/{i}.png"
            Image.frombytes("RGB", (32, 32), bytes(rng.randrange(256) for _ in range(3072))).save(
                path
            )
            self.samples.append(
                dict(
                    image=f"images/{i}.png",
                    label=i % 2,
                    sha256=sha(path.read_bytes()),
                    mime="image/png",
                    source_dataset="fixture",
                )
            )
        self.split = {
            "version": 2,
            "pool_per_class": 4,
            "seed": 42,
            "reference_pool": [s["image"] for s in self.samples[:8]],
            "references": [s["image"] for s in self.samples[:4]],
            "test": [s["image"] for s in self.samples[8:]],
            "samples": {
                s["image"]: dict(
                    label=s["label"],
                    sha256=s["sha256"],
                    group_id=f"g{i}",
                    provenance_group_keys=[f"patient{i}"],
                )
                for i, s in enumerate(self.samples)
            },
        }
        self.reference_set = FixedReferences(self.root, self.samples, self.split, 8)
        self.query = Query.from_sample(self.samples[8], self.root, self.split)
        self.library = KnowledgeLibrary(LAB / "knowledge")
        self.prompts = {
            "basic": (PROJECT / "prompts/he_binary.txt").read_text(),
            **{n: (LAB / f"prompts/{n}.txt").read_text() for n in ("observe", "refine")},
        }

    def infer(self, client, **kwargs):
        kwargs.setdefault("mode", "knowledge_review")
        return infer_one(
            self.query, self.reference_set, self.library, client, self.prompts, **kwargs
        )

    def test_fixed_references_match_v1_for_all_counts(self):
        for count in (4, 6, 8):
            fixed = FixedReferences(self.root, self.samples, self.split, count)
            expected = select_references(self.split, self.samples, count)
            refs, log = fixed.prepare(self.query)
            self.assertEqual(
                [r["image"] for r in log["references"]], [r["image"] for r in expected]
            )
            self.assertEqual(sum(r["label"] for r in refs), count // 2)
            self.assertEqual(refs, fixed.prepare(self.query)[0])
            for ref, meta in zip(refs, expected):
                self.assertEqual(ref["bytes"], (self.root / meta["image"]).read_bytes())
        self.assertNotIn("label", vars(self.query))

    def test_same_patient_fails_without_substitution(self):
        q = Query(
            self.query.image,
            self.query.path,
            self.query.mime,
            self.query.sha256,
            "g6",
            ("patient7",),
        )
        with self.assertRaises(UnsafeReferences):
            self.reference_set.prepare(q)

    def test_pixels_and_near_duplicate_fail_without_substitution(self):
        fp = fingerprint(self.query.path)
        for key in ("pixel_sha256", "dhash"):
            fixed = FixedReferences(self.root, self.samples, self.split, 8)
            fixed.rows[0][key] = fp[key]
            with self.assertRaises(UnsafeReferences):
                fixed.prepare(self.query)

    def test_invalid_count_and_metadata_rejected(self):
        for count in (3, 10):
            with self.assertRaises(ValueError):
                FixedReferences(self.root, self.samples, self.split, count)
        split = copy.deepcopy(self.split)
        split["samples"]["images/0.png"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            FixedReferences(self.root, self.samples, split, 8)

    def test_default_review_has_no_knowledge_dependency(self):
        args = parser().parse_args(["run"])
        self.assertEqual((args.mode, args.reference_count), ("review", 8))
        client = Client([response(observation()), response({"p_he": 0.6, "evidence": "可见"})])
        result = infer_one(self.query, self.reference_set, None, client, self.prompts)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["card_ids"], [])
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0].content, client.requests[1].content)

    def test_reference_drift_during_run_fatal(self):
        (self.root / "images/6.png").write_bytes(b"changed")
        result = self.infer(Client([]))
        self.assertTrue(result["fatal"])
        self.assertEqual(result["error"], "stale_data")
        self.assertEqual(result["calls"], [])

    def test_query_drift_fatal(self):
        # 保持仍然可解码，测试 SHA 校验而非图像解码异常。
        Image.new("RGB", (32, 32), "red").save(self.query.path)
        result = self.infer(Client([]))
        self.assertTrue(result["fatal"])

    def test_low_probability_still_review_no_anchoring(self):
        client = Client(
            [
                response(observation(0.0123456789)),
                response({"p_he": 0.8, "evidence": "复核"}),
            ]
        )
        result = self.infer(client)
        self.assertEqual(result["p_he"], 0.8)
        self.assertEqual(len(client.requests), 2)
        second = client.requests[1]
        self.assertNotIn("0.0123456789", second.system_prompt)
        self.assertNotIn("STAGE1_EVIDENCE_SENTINEL", repr(second))
        self.assertEqual(client.requests[0].content, second.content)
        self.assertEqual(len(result["card_ids"]), 2)
        self.assertEqual(len([p for p in second.content if isinstance(p, ImagePart)]), 9)

    def test_nonfatal_stage1_failures_review(self):
        for first in (
            ModelError("API_SECRET"),
            response({"wrong": True}),
            response(observation(), False),
            response(observation(None)),
        ):
            with self.subTest(first=first):
                client = Client([first, response({"p_he": 0.4, "evidence": "可见"})])
                result = self.infer(client)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(len(client.requests), 2)
                self.assertNotIn("API_SECRET", json.dumps(result))

    def test_fatal_stage1_stops(self):
        result = self.infer(Client([ModelError("API_SECRET", fatal=True)]))
        self.assertTrue(result["fatal"])
        self.assertEqual(len(result["calls"]), 1)
        self.assertIsNone(result["stage2"])

    def test_no_substitute_for_failed_stage2(self):
        for final, status in (
            (ModelError("failed"), "error"),
            (response({"p_he": None, "evidence": "不可判读"}), "abstain"),
            (response({"p_he": 0.9, "evidence": "truncated"}, False), "error"),
        ):
            result = self.infer(Client([response(observation(0.9)), final]))
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["p_he"])

    def test_three_modes_budget_and_no_knowledge_control(self):
        for mode in ("reference", "review", "knowledge_review"):
            values = [] if mode == "reference" else [response(observation())]
            client = Client(values + [response({"p_he": 0.6, "evidence": "可见"})])
            result = self.infer(client, mode=mode)
            self.assertEqual(len(result["calls"]), 1 if mode == "reference" else 2)
            if mode != "knowledge_review":
                self.assertFalse(result["card_ids"])
                self.assertNotIn("[C01]", client.requests[-1].system_prompt)

    def test_truth_permutation_does_not_change_inference(self):
        first = Client([response(observation()), response({"p_he": 0.4, "evidence": "可见"})])
        result1 = self.infer(first)
        samples = copy.deepcopy(self.samples)
        split = copy.deepcopy(self.split)
        for sample in samples[8:]:
            sample["label"] = 1 - sample["label"]
            split["samples"][sample["image"]]["label"] = sample["label"]
        self.query = Query.from_sample(samples[8], self.root, split)
        second = Client([response(observation()), response({"p_he": 0.4, "evidence": "可见"})])
        result2 = self.infer(second)
        self.assertEqual(first.requests, second.requests)
        for key in ("references", "branches", "card_ids", "p_he"):
            self.assertEqual(result1[key], result2[key])

    def test_schema_rejects_unsafe_fields(self):
        for change in (
            {"p_he": True},
            {"p_he": 2},
            {"p_he": float("nan")},
            {"quality": "x"},
            {"appearance_tags": ["CE5"]},
            {"appearance_tags": ["cystic", "cystic"]},
            {"observations": []},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_observation(response(observation() | change))
        with self.assertRaises(ValueError):
            parse_observation(ModelResponse('{"p_he":0.2,"p_he":0.4}', True))

    def test_route_does_not_use_probability(self):
        for probability in (0, 0.1, 0.99, None):
            self.assertEqual(
                route(observation(probability, ["mixed", "solid", "calcified"])),
                ["mixed_solid", "calcified"],
            )
        self.assertEqual(route(observation(tags=["no_focal_lesion", "cystic"])), ["general"])
        self.assertEqual(route(None), ["general"])

    def test_card_budget_empty_and_checksums(self):
        for branches in (["cystic"], ["mixed_solid", "calcified"], ["general"]):
            cards = self.library.retrieve(branches)
            self.assertLessEqual(len(cards), 2)
            self.assertLessEqual(len("\n".join(self.library.render(c) for c in cards)), 600)
        self.assertEqual(self.library.retrieve(["does_not_exist"]), [])
        self.assertEqual(self.library.retrieve(["cystic"], budget=1), [])
        with self.assertRaises(ValueError):
            self.library.retrieve(["cystic"], limit=3)
        self.library.cards = []
        result = self.infer(
            Client([response(observation()), response({"p_he": 0.6, "evidence": "可见"})])
        )
        self.assertEqual(result["card_ids"], [])
        self.assertEqual(result["status"], "ok")

    def test_dryrun_no_calls_and_truth_absent(self):
        client = Client([])
        result = self.infer(client, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["calls"], [])
        self.assertNotIn("true_label", json.dumps(result))
        self.assertEqual(result["dry_request"]["image_parts"], 9)

    def test_paired_metrics(self):
        base = dict(
            image="a",
            source_dataset="x",
            branches=[],
            calls=[],
            latency_seconds=0,
            status="ok",
            true_label=1,
            stage1={"status": "ok", "prediction": {"p_he": 0.1}},
        )
        rows = [dict(base, p_he=0.8), dict(base, image="b", true_label=0, p_he=0.8)]
        metrics = evaluate(rows, [dict(image=r["image"], source_dataset="x") for r in rows])
        self.assertEqual(
            metrics["paired_stage1_stage2"]["changes"],
            {"corrections": 1, "regressions": 1},
        )
        self.assertAlmostEqual(metrics["brier_score_on_valid"], 0.34)

    def test_both_adapters_all_modes_and_original_bytes(self):
        for provider in ("glm", "openai"):
            for mode in ("reference", "review", "knowledge_review"):
                client = create_model(provider)
                values = [] if mode == "reference" else [observation()]
                values += [{"p_he": 0.7, "evidence": "可见"}]
                raw = []
                for value in values:
                    if provider == "glm":
                        raw.append(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "message": {
                                            "content": json.dumps(value),
                                            "reasoning_content": "HIDDEN",
                                        },
                                    }
                                ]
                            }
                        )
                    else:
                        raw.append(
                            {
                                "status": "completed",
                                "output": [
                                    {"type": "reasoning", "summary": []},
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "status": "completed",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": json.dumps(value),
                                            }
                                        ],
                                    },
                                ],
                            }
                        )
                with patch.object(client, "_post", side_effect=raw) as post:
                    result = self.infer(client, mode=mode)
                with self.subTest(provider=provider, mode=mode):
                    self.assertEqual(result["status"], "ok")
                    self.assertNotIn("HIDDEN", json.dumps(result))
                    self.assertEqual(post.call_count, 1 if mode == "reference" else 2)
                    import base64

                    query64 = base64.b64encode(self.query.path.read_bytes()).decode()
                    payload = json.dumps(post.call_args.args[0])
                    self.assertIn(query64, payload)
                    self.assertNotIn("true_label", payload)
                    self.assertNotIn(self.query.image, payload)

    def test_cli_outputs_and_fatal_unattempted(self):
        (self.root / "labels/labels.txt").write_text(
            "".join(f"{s['image']} {s['label']}\n" for s in self.samples)
        )
        (self.root / "labels/manifest.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in self.samples)
        )
        splitpath = self.root / "split.json"
        splitpath.write_text(json.dumps(self.split))
        args = parser().parse_args(
            [
                "run",
                "--dataset",
                str(self.root),
                "--split",
                str(splitpath),
                "--output",
                str(self.root / "output"),
            ]
        )
        client = Client([ModelError("SECRET", fatal=True)])
        with patch("builtins.print"):
            output = run(args, client)
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["unattempted"], 1)
        self.assertEqual(json.loads((output / "run.json").read_text())["status"], "fatal_stopped")
        for name in ("references", "stage1", "stage2", "trace", "predictions"):
            self.assertTrue((output / f"{name}.jsonl").exists())
            self.assertNotIn("SECRET", (output / f"{name}.jsonl").read_text())


class TransportTests(unittest.TestCase):
    def test_actual_urllib_attempts_two_stages_six_requests(self):
        values = [observation(), {"p_he": 0.7, "evidence": "可见"}]

        class Handler(urllib.request.HTTPSHandler):
            def __init__(self):
                super().__init__()
                self.count = 0

            def https_open(self, request):
                self.count += 1
                if self.count % 3:
                    raise urllib.error.URLError("SECRET transient")
                obj = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(values[self.count // 3 - 1])},
                        }
                    ],
                    "usage": {"total_tokens": 12},
                }
                result = urllib.response.addinfourl(
                    io.BytesIO(json.dumps(obj).encode()), {}, request.full_url, 200
                )
                result.msg = "OK"
                return result

        handler = Handler()
        opener = urllib.request.build_opener(handler)
        client = MeteredClient(create_model("glm", retries=2), retries=2)
        request = ModelRequest("safe", (TextPart("safe"),))
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": "SECRET"}),
            patch("models.glm.urllib.request.build_opener", return_value=opener),
            patch("models.glm.time.sleep"),
        ):
            client.generate(request)
            client.generate(request)
        self.assertEqual([c["http_attempts"] for c in client.calls], [3, 3])
        self.assertEqual(handler.count, 6)
        self.assertNotIn("SECRET", json.dumps(client.calls))
        with self.assertRaises(ModelError):
            client.generate(request)

    def test_http_hard_cap(self):
        class BadClient:
            def generate(self, request):
                for _ in range(10):
                    sys.audit("urllib.Request", "ignored")

        client = MeteredClient(BadClient(), retries=2)
        with self.assertRaises(ModelError):
            client.generate(None)
        self.assertEqual(client.calls[0]["http_attempts"], 3)


if __name__ == "__main__":
    unittest.main()
