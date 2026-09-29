"""对三家真实适配器的十二组配置做完整模拟，检查图像、提示词与真值隔离。"""

import base64
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
from test_openai import response_body

import he_agent as agent
from models import create_model
from prompt_profiles import KNOWLEDGE_MODES, MODES, REFERENCE_MODES, load_prompt


class MatrixTests(unittest.TestCase):
    def fixture(self, root, pool_size=4):
        """建立可配置大小的参考池及两张测试图，路径和来源为隔离检查哨兵。"""
        (root / "images").mkdir()
        (root / "labels").mkdir()
        rows = []
        for i in range(pool_size + 2):
            path = root / "images" / f"private_case_{i}.png"
            Image.new("L", (8, 8), color=i).save(path)
            rows.append(
                {
                    "image": str(path.relative_to(root)),
                    "label": i % 2,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_dataset": "private_source_marker",
                }
            )
        (root / "labels/labels.txt").write_text(
            "".join(f"{r['image']} {r['label']}\n" for r in rows)
        )
        (root / "labels/manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        split = {
            "reference_pool": [r["image"] for r in rows[:pool_size]],
            "references": [r["image"] for r in rows[:4]],
            "test": [r["image"] for r in rows[pool_size:]],
            "samples": {r["image"]: {**r, "group_id": str(i)} for i, r in enumerate(rows)},
        }
        path = root / "split.json"
        path.write_text(json.dumps(split))
        return rows, path

    def test_all_providers_four_modes_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, split = self.fixture(root)
            selected_lists = []
            for provider in ("glm", "openai", "deepseek"):
                for mode in MODES:
                    with self.subTest(provider=provider, mode=mode):
                        args = SimpleNamespace(
                            dataset=root,
                            split=split,
                            mode=mode,
                            prompt=None,
                            provider=provider,
                            model=None,
                            output=root / f"{provider}_{mode}",
                            limit=0,
                            seed=42,
                            max_tokens=8192,
                            timeout=1,
                            retries=0,
                            dry_run=False,
                            threshold=0.5,
                        )
                        client = create_model(provider)
                        body = (
                            response_body()
                            if provider == "openai"
                            else {
                                "choices": [
                                    {
                                        "message": {
                                            "content": '{"p_he":0.73,"evidence":"visible structures"}'
                                        },
                                        "finish_reason": "stop",
                                    }
                                ]
                            }
                        )
                        with (
                            patch.object(client, "validate_credentials"),
                            patch.object(client, "_post", return_value=body) as post,
                        ):
                            self.assertEqual(agent.run(args, model_client=client), 0)
                        samples = json.loads((args.output / "samples.json").read_text())
                        selected_lists.append(samples)
                        self.assertEqual(post.call_count, 2)
                        for index, call in enumerate(post.call_args_list):
                            payload = call.args[0]
                            prompt = (
                                payload["instructions"]
                                if provider == "openai"
                                else payload["messages"][0]["content"]
                            )
                            parts = (
                                payload["input"][0]["content"]
                                if provider == "openai"
                                else payload["messages"][1]["content"]
                            )
                            self.assertEqual(prompt, load_prompt(mode)[0])
                            text = " ".join(
                                p["text"] for p in parts if p["type"] in ("text", "input_text")
                            )
                            image_parts = [
                                p for p in parts if p["type"] in ("image_url", "input_image")
                            ]
                            expected = (
                                [root / r["image"] for r in rows[:4]]
                                if mode in REFERENCE_MODES
                                else []
                            ) + [root / samples[index]["image"]]
                            self.assertEqual(len(image_parts), len(expected))
                            self.assertEqual(
                                text.count("诊断标签"), 4 if mode in REFERENCE_MODES else 0
                            )
                            for part, path in zip(image_parts, expected):
                                url = (
                                    part["image_url"]
                                    if provider == "openai"
                                    else part["image_url"]["url"]
                                )
                                self.assertEqual(
                                    base64.b64decode(url.split(",", 1)[1]), path.read_bytes()
                                )
                            self.assertNotIn("private_case_", json.dumps(payload))
                            self.assertNotIn("private_source_marker", json.dumps(payload))
                        config = json.loads((args.output / "run.json").read_text())
                        self.assertEqual(config["uses_references"], mode in REFERENCE_MODES)
                        self.assertEqual(config["uses_knowledge"], mode in KNOWLEDGE_MODES)
                        refs = json.loads((args.output / "references.json").read_text())
                        self.assertEqual(len(refs), 4 if mode in REFERENCE_MODES else 0)
                        self.assertEqual(
                            (args.output / "knowledge_sources.json").exists(),
                            mode in KNOWLEDGE_MODES,
                        )
                        records = [
                            json.loads(line)
                            for line in (args.output / "predictions.jsonl").read_text().splitlines()
                        ]
                        self.assertTrue(
                            all(r["p_he"] == 0.73 and r["status"] == "ok" for r in records)
                        )
                        self.assertTrue(all(abs(r["p_non_he"] - 0.27) < 1e-12 for r in records))
                        self.assertEqual(
                            json.loads((args.output / "metrics.json").read_text())["coverage"], 1
                        )
            self.assertTrue(all(samples == selected_lists[0] for samples in selected_lists))

    def test_expanded_references_in_all_adapters_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = self.fixture(root, pool_size=8)
            sample_lists = []
            for provider in ("glm", "openai", "deepseek"):
                for mode in sorted(REFERENCE_MODES):
                    for count in (6, 8):
                        args = SimpleNamespace(
                            dataset=root,
                            split=split,
                            mode=mode,
                            reference_count=count,
                            prompt=None,
                            provider=provider,
                            model=None,
                            output=root / f"{provider}_{mode}_{count}",
                            limit=0,
                            seed=42,
                            max_tokens=8192,
                            timeout=1,
                            retries=0,
                            dry_run=False,
                        )
                        client = create_model(provider)
                        body = (
                            response_body()
                            if provider == "openai"
                            else {
                                "choices": [
                                    {
                                        "message": {
                                            "content": '{"p_he":0.73,"evidence":"visible structures"}'
                                        },
                                        "finish_reason": "stop",
                                    }
                                ]
                            }
                        )
                        with (
                            patch.object(client, "validate_credentials"),
                            patch.object(client, "_post", return_value=body) as post,
                        ):
                            self.assertEqual(agent.run(args, model_client=client), 0)
                        self.assertEqual(post.call_count, 2)
                        samples = json.loads((args.output / "samples.json").read_text())
                        sample_lists.append(samples)
                        refs = json.loads((args.output / "references.json").read_text())
                        self.assertEqual(len(refs), count)
                        self.assertEqual(sum(r["label"] for r in refs), count // 2)
                        config = json.loads((args.output / "run.json").read_text())
                        self.assertEqual(config["reference_count"], count)
                        for i, call in enumerate(post.call_args_list):
                            payload = call.args[0]
                            parts = (
                                payload["input"][0]["content"]
                                if provider == "openai"
                                else payload["messages"][1]["content"]
                            )
                            images = [p for p in parts if p["type"] in ("image_url", "input_image")]
                            texts = " ".join(
                                p["text"] for p in parts if p["type"] in ("text", "input_text")
                            )
                            self.assertEqual(len(images), count + 1)
                            self.assertEqual(texts.count("诊断标签"), count)
                            for part, row in zip(images, refs + [samples[i]]):
                                url = (
                                    part["image_url"]
                                    if provider == "openai"
                                    else part["image_url"]["url"]
                                )
                                self.assertEqual(
                                    base64.b64decode(url.split(",", 1)[1]),
                                    (root / row["image"]).read_bytes(),
                                )
                            self.assertNotIn("private_case_", json.dumps(payload))
                            self.assertNotIn("private_source_marker", json.dumps(payload))
            self.assertTrue(all(samples == sample_lists[0] for samples in sample_lists))
            for mode in ("baseline", "knowledge"):
                args.mode = mode
                args.output = root / f"invalid_{mode}"
                with self.assertRaisesRegex(ValueError, "--reference-count requires"):
                    agent.run(args)
                self.assertFalse(args.output.exists())

    def test_combined_requires_split_and_frozen_medical_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = self.fixture(root)
            args = SimpleNamespace(
                dataset=root,
                mode="reference_knowledge",
                split=None,
                prompt=None,
                limit=0,
                max_tokens=8192,
                timeout=1,
                retries=0,
                output=root / "invalid",
            )
            with self.assertRaisesRegex(ValueError, "require --split"):
                agent.run(args)
            self.assertFalse(args.output.exists())
            args.split = split
            with self.assertRaises(ValueError):
                load_prompt(
                    "reference_knowledge",
                    Path(__file__).resolve().parents[1] / "prompts/he_binary.txt",
                )
            with self.assertRaises(ValueError):
                load_prompt("unknown")
            prompt, metadata, sources = load_prompt("reference_knowledge")
            self.assertEqual((prompt, metadata, sources), load_prompt("knowledge"))
            import prompt_profiles

            fake_project = root / "fake"
            (fake_project / "prompts").mkdir(parents=True)
            for filename in ("he_medical.txt", "he_medical.sources.json", "he_binary.txt"):
                (fake_project / "prompts" / filename).write_bytes(
                    (prompt_profiles.PROJECT / "prompts" / filename).read_bytes()
                )
            (fake_project / "prompts/he_medical.txt").write_text("tampered")
            with (
                patch.object(prompt_profiles, "PROJECT", fake_project),
                self.assertRaises(ValueError),
            ):
                load_prompt("reference_knowledge")

    def test_all_dry_runs_need_no_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split = self.fixture(root)
            for provider in ("glm", "openai", "deepseek"):
                for mode in MODES:
                    args = SimpleNamespace(
                        dataset=root,
                        mode=mode,
                        split=split,
                        prompt=None,
                        limit=0,
                        max_tokens=8192,
                        timeout=1,
                        retries=0,
                        output=root / f"{provider}_{mode}",
                        seed=42,
                        model=None,
                        dry_run=True,
                    )
                    client = create_model(provider)
                    with (
                        patch.object(
                            client, "generate", side_effect=AssertionError("Network attempted")
                        ),
                        patch.object(
                            client,
                            "validate_credentials",
                            side_effect=AssertionError("Credential check attempted"),
                        ),
                    ):
                        self.assertEqual(agent.run(args, model_client=client), 0)
                    self.assertFalse((args.output / "predictions.jsonl").exists())
                    self.assertEqual(
                        json.loads((args.output / "validation.json").read_text())["network_calls"],
                        0,
                    )


if __name__ == "__main__":
    unittest.main()
