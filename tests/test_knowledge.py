"""验证医学提示词冻结校验及知识模式的数据隔离。"""

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
import he_agent as agent
import prompt_profiles
from models import ImagePart, ModelResponse

PROJECT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self):
        self.requests = []

    def describe(self):
        return {"provider": "test", "model": "fake"}

    def validate_credentials(self):
        pass

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(text='{"p_he":null,"evidence":"无法辨认关键结构"}', completed=True)


class KnowledgeTests(unittest.TestCase):
    def test_frozen_prompt_and_evidence(self):
        text, meta, sources = prompt_profiles.load_prompt("knowledge")
        self.assertEqual(meta["prompt_profile"], "medical-v3-soft-guidance-probability")
        self.assertEqual(len(sources["sources"]), 11)
        self.assertEqual(
            {r["rule_id"] for r in sources["rules"]}, {"K%02d" % i for i in range(1, 9)}
        )
        self.assertTrue(text.startswith((PROJECT / "prompts/he_binary.txt").read_text()))
        self.assertNotIn("images/", text)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), sources["prompt_sha256"])

    def test_baseline_and_reference_use_probability_prompt(self):
        baseline = (PROJECT / "prompts/he_binary.txt").read_text()
        for mode in ["baseline", "reference"]:
            text, meta, sources = prompt_profiles.load_prompt(mode)
            self.assertEqual(text, baseline)
            self.assertIsNone(sources)
            self.assertEqual(meta["prompt_profile"], "baseline-probability-v2")

    def test_tamper_and_wrong_profile_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "prompts").mkdir()
            for filename in [
                "he_binary.txt",
                "he_medical.txt",
                "he_medical.sources.json",
            ]:
                (root / "prompts" / filename).write_bytes(
                    (PROJECT / "prompts" / filename).read_bytes()
                )
            (root / "prompts/he_medical.txt").write_text("modified")
            with patch.object(prompt_profiles, "PROJECT", root):
                with self.assertRaises(ValueError):
                    prompt_profiles.load_prompt("knowledge")
            with self.assertRaises(ValueError):
                prompt_profiles.load_prompt("knowledge", root / "prompts/he_binary.txt")

    def test_knowledge_workflow_sends_one_test_image_without_references(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            rows = []
            for i in range(6):
                f = root / "images" / f"{i}.png"
                Image.new("L", (8, 8), color=i).save(f)
                rows.append(
                    {
                        "image": f"images/{i}.png",
                        "label": i % 2,
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
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
                output=root / "knowledge",
                mode="knowledge",
                split=sp,
                prompt=None,
                limit=0,
                seed=42,
                model=None,
                provider="glm",
                max_tokens=8192,
                timeout=1,
                retries=0,
                dry_run=False,
            )
            client = FakeModel()
            self.assertEqual(agent.run(args, model_client=client), 0)
            self.assertEqual(len(client.requests), 2)
            for r in client.requests:
                self.assertEqual(sum(isinstance(x, ImagePart) for x in r.content), 1)
                self.assertIn("K04", r.system_prompt)
                self.assertNotIn("诊断标签", str(r.content))
            self.assertEqual(json.loads((args.output / "references.json").read_text()), [])
            self.assertTrue((args.output / "knowledge_sources.json").exists())
            self.assertEqual(
                json.loads((args.output / "metrics.json").read_text())["abstentions"], 2
            )
            knowledge_samples = json.loads((args.output / "samples.json").read_text())
            args.mode = "baseline"
            args.output = root / "baseline"
            base = FakeModel()
            agent.run(args, model_client=base)
            self.assertEqual(
                knowledge_samples, json.loads((args.output / "samples.json").read_text())
            )
            self.assertEqual(
                [r.content for r in client.requests], [r.content for r in base.requests]
            )
            self.assertNotEqual(client.requests[0].system_prompt, base.requests[0].system_prompt)
            args.mode = "knowledge"
            args.split = None
            args.output = root / "invalid"
            with self.assertRaises(ValueError):
                agent.run(args, model_client=FakeModel())
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
