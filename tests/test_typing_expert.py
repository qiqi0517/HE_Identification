"""Expert isolation, frozen encoders, provenance and selection integration."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_TORCH = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchvision", "timm"))
if HAS_TORCH:
    import torch
    from PIL import Image
    from torch import nn

    from typing_expert import CLASSES
    from typing_expert.data import code_hash, load_split, prepare, read, rows_for, sha, write
    from typing_expert.metrics import paired_comparison, score
    from typing_expert.network import PREPROCESS, Expert
    from typing_expert.options import Standardize, class_weights, resolve_options, stopping_key
    from typing_expert.training import (
        checkpoint,
        feature_cache,
        load_checkpoint,
        prediction,
        tensor_hash,
        train,
    )
    from typing_expert.workflow import evaluate, select

    class ToyExpert(nn.Module):
        """Cheap encoder for workflow tests, never used in actual experiments."""

        def __init__(self, variant, **kwargs):
            super().__init__()
            self.variant = variant
            self.dimension = 4
            self.pretrained_hash = "fixture-encoder"
            self.encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))
            with torch.no_grad():
                self.encoder[2].weight.fill_(0.1)
                self.encoder[2].bias.zero_()
            self.encoder.requires_grad_(False)
            self.head = nn.Linear(4, 7)

        def forward(self, x):
            return self.head(self.encoder(x))

        def phase(self, finetune=False):
            self.encoder.eval().requires_grad_(False)
            self.head.train()
            return [{"params": self.head.parameters(), "lr": 1e-3}]


@unittest.skipUnless(HAS_TORCH, "Install requirements-expert.txt to run expert tests")
class ExpertTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data"
        (self.dataset / "images").mkdir(parents=True)
        (self.dataset / "labels").mkdir()
        manifest, labels, samples, partitions = [], [], {}, {}
        for i, role in enumerate(("reference", "search", "selection", "test")):
            partitions[role] = []
            for j, label in enumerate(CLASSES):
                name = f"images/{role}-{label}.png"
                Image.new("RGB", (28, 24), (10 + i * 40, j * 30, 50)).save(self.dataset / name)
                digest = sha(self.dataset / name)
                partitions[role].append(name)
                samples[name] = dict(
                    label=label, sha256=digest, group_id=name, source_dataset="fixture", role=role
                )
                manifest.append(
                    dict(
                        image=name,
                        typing_label=label,
                        primary_label_eligible=True,
                        sha256=digest,
                        sources=["fixture"],
                    )
                )
                labels.append(f"{name} {label}\n")
        (self.dataset / "labels/manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in manifest)
        )
        (self.dataset / "labels/labels_typing.txt").write_text("".join(labels))
        self.source = self.root / "source.json"
        write(
            self.source,
            dict(
                schema="he-typing-split-v1",
                partitions=partitions,
                samples=samples,
                dataset_labels_sha256=sha(self.dataset / "labels/labels_typing.txt"),
            ),
        )
        self.split_path = self.root / "split.json"
        prepare(self.dataset, self.source, self.split_path)
        self.split, _ = load_split(self.split_path)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, output, **kwargs):
        values = dict(
            output=output,
            split=self.split_path,
            dataset=None,
            variant="S1",
            seed=42,
            device="cpu",
            threads=1,
            tiny_weights=None,
            cache=self.root / "cache",
            smoke=False,
            epochs=None,
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_metadata_prepare_and_train_never_read_test_pixels(self):
        for name in self.split["partitions"]["test"]:
            (self.dataset / name).unlink()
        load_split(self.split_path)  # metadata-only access
        with patch("typing_expert.training.Expert", ToyExpert):
            result = train(self.args(self.root / "train", epochs=2))
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["protocol_eligible"])
        self.assertTrue(result["frozen_parameters_verified"])

    def test_split_rejects_groups_labels_and_membership_changes(self):
        split = read(self.source)
        a, b = split["partitions"]["search"][0], split["partitions"]["test"][0]
        split["samples"][b]["group_id"] = split["samples"][a]["group_id"]
        write(self.source, split)
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare(self.dataset, self.source, self.root / "bad.json")
        mapped = read(self.split_path)
        mapped["partitions"]["train"].append(mapped["partitions"]["test"].pop())
        write(self.root / "tamper.json", mapped)
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_split(self.root / "tamper.json")
        (self.dataset / "labels/labels_typing.txt").write_text("bad CE1\n")
        with self.assertRaises((ValueError, KeyError)):
            load_split(self.split_path)

    def test_cache_reuse_corruption_and_source_change(self):
        rows = rows_for(self.split, "train")[:2]
        model = ToyExpert("S1")
        device = torch.device("cpu")
        cache = self.root / "features"
        first = feature_cache(model, self.dataset, rows, device, cache, {"version": 1})
        with patch.object(model.encoder, "forward", side_effect=AssertionError("cache miss")):
            second = feature_cache(model, self.dataset, rows, device, cache, {"version": 1})
        self.assertTrue(torch.equal(first, second))
        p = next(cache.glob("*.pt"))
        payload = torch.load(p, weights_only=True)
        payload["features"][0, 0] += 1
        torch.save(payload, p)
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            feature_cache(model, self.dataset, rows, device, cache, {"version": 1})
        Image.new("RGB", (28, 24), "white").save(self.dataset / rows[0]["image"])
        with self.assertRaisesRegex(ValueError, "changed"):
            feature_cache(model, self.dataset, rows, device, cache, {"version": 2})

    def test_checkpoint_class_order_and_implementation_guard(self):
        config = dict(variant="S1", preprocess=PREPROCESS, code_sha256=code_hash())
        model = ToyExpert("S1").eval()
        x = torch.ones(1, 3, 224, 224)
        path = self.root / "model.pt"
        torch.save(checkpoint(model, config, 1), path)
        with patch("typing_expert.training.Expert", ToyExpert):
            restored, _ = load_checkpoint(path, torch.device("cpu"))
        self.assertTrue(torch.equal(model(x), restored(x)))
        state = torch.load(path, weights_only=True)
        state["classes"].reverse()
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "class order"):
            load_checkpoint(path, torch.device("cpu"))

    def test_actual_resnet_batchnorm_and_encoder_frozen(self):
        model = Expert("S2")  # architecture only; no network download
        before = {k: tensor_hash(v) for k, v in model.encoder.state_dict().items()}
        opt = torch.optim.AdamW(model.phase())
        model(torch.randn(2, 3, 224, 224)).sum().backward()
        opt.step()
        self.assertTrue(
            all(tensor_hash(v) == before[k] for k, v in model.encoder.state_dict().items())
        )
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_actual_tiny_unfreezes_only_last_two_blocks_and_norm(self):
        model = Expert("S3")
        model.phase(False)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        model.phase(True)
        for name, p in model.encoder.named_parameters():
            self.assertEqual(
                p.requires_grad, name.startswith(("blocks.10.", "blocks.11.", "norm."))
            )
        loss = model(torch.randn(1, 3, 224, 224)).square().sum()
        loss.backward()
        self.assertIsNotNone(model.encoder.blocks[-1].attn.qkv.weight.grad)
        self.assertIsNone(model.encoder.patch_embed.proj.weight.grad)

    def test_fixed_denominator_and_probability_metrics(self):
        rows = rows_for(self.split, "test")[:3]
        pred = [
            dict(image=rows[0]["image"], **prediction(torch.tensor([10.0, 0, 0, 0, 0, 0, 0]))),
            dict(image=rows[1]["image"], status="error", predicted_type=None, confidence=None),
        ]
        metric = score(rows, pred)
        self.assertEqual(metric["accuracy"], 1 / 3)
        self.assertEqual(metric["unattempted"], 1)
        self.assertEqual(metric["errors"], 1)
        self.assertEqual(metric["probability_count"], 1)
        self.assertFalse(metric["complete"])
        with self.assertRaises(ValueError):
            score(rows, pred + pred[:1])
        with self.assertRaises(ValueError):
            prediction(torch.full((7,), float("nan")))

    def test_selection_and_frozen_test_full_workflow(self):
        runs = self.root / "runs"
        with patch("typing_expert.training.Expert", ToyExpert):
            train(self.args(runs / "S1" / "42"))
        source = runs / "S1" / "42"
        # Fixture copies isolate tie-breaking from expensive nine-run training.
        import shutil

        for variant in ("S1", "S2", "S3"):
            for seed in (42, 43, 44):
                dest = runs / variant / str(seed)
                if dest == source:
                    continue
                shutil.copytree(source, dest)
                config = read(dest / "config.json")
                config.update(variant=variant, seed=seed)
                write(dest / "config.json", config)
                payload = torch.load(dest / "best.pt", weights_only=True)
                payload["config"] = config
                torch.save(payload, dest / "best.pt")
                status = read(dest / "status.json")
                status["checkpoint_sha256"] = sha(dest / "best.pt")
                write(dest / "status.json", status)
        freeze = self.root / "selection.json"
        result = select(runs, freeze)
        self.assertEqual((result["selected"]["variant"], result["selected"]["seed"]), ("S1", 42))
        args = SimpleNamespace(
            output=self.root / "eval",
            freeze=freeze,
            variant=None,
            seed=42,
            device="cpu",
            threads=1,
            dataset=None,
        )
        with patch("typing_expert.training.Expert", ToyExpert):
            evaluated = evaluate(args)
        self.assertEqual(evaluated["planned"], 7)
        self.assertEqual(read(args.output / "metrics.json")["attempted"], 7)
        manifest = self.root / "candidates.json"
        write(
            manifest, {"candidates": [{"name": "S1", "variant": "S1", "expected": {"epochs": 50}}]}
        )
        custom_freeze = self.root / "custom_freeze.json"
        self.assertEqual(select(runs, custom_freeze, manifest)["selected"]["candidate"], "S1")
        write(
            manifest, {"candidates": [{"name": "S1", "variant": "S1", "expected": {"epochs": 200}}]}
        )
        with self.assertRaisesRegex(ValueError, "declared candidate"):
            select(runs, self.root / "bad_freeze.json", manifest)
        (source / "best.pt").write_bytes(b"changed")
        args.output = self.root / "eval_tampered"
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate(args)

    def test_smoke_rejected_for_selection_and_outputs_not_overwritten(self):
        run = self.root / "runs/S1/42"
        with patch("typing_expert.training.Expert", ToyExpert):
            train(self.args(run, smoke=True, epochs=1))
        with self.assertRaisesRegex(ValueError, "smoke"):
            select(self.root / "runs", self.root / "freeze.json")
        with self.assertRaises(FileExistsError):
            train(self.args(run))

    def test_paired_group_comparison(self):
        rows = rows_for(self.split, "test")
        correct = [
            dict(image=r["image"], status="ok", predicted_type=r["label"], confidence=0.8)
            for r in rows
        ]
        worse = copy.deepcopy(correct)
        worse[0].update(predicted_type="CE1")
        metric = paired_comparison(rows, correct, worse, repeats=100)
        self.assertAlmostEqual(metric["first_minus_second"], 1 / 7)
        with self.assertRaises(ValueError):
            paired_comparison(rows, correct, worse[:-1])

    def test_standardization_uses_train_only_and_survives_checkpoint(self):
        output = self.root / "scaled"
        with patch("typing_expert.training.Expert", ToyExpert):
            train(
                self.args(
                    output,
                    feature_scaling="standard",
                    epochs=2,
                    early_stop_metric="loss",
                    patience=3,
                )
            )
            model, _ = load_checkpoint(output / "best.pt", torch.device("cpu"))
        self.assertIsInstance(model.head[0], Standardize)
        fitted = read(output / "feature_scaling.json")
        self.assertEqual(fitted["count"], 14)
        from typing_expert.training import Images

        raw = torch.stack(
            [Images(self.dataset, rows_for(self.split, "train"))[i][0] for i in range(14)]
        )
        with torch.no_grad():
            features = model.encoder(raw)
            expected = features.double().mean(0).float()
            torch.testing.assert_close(model.head[0].mean, expected)
            torch.testing.assert_close(model(raw), model.head(features))
        self.assertEqual(read(output / "config.json")["early_stop_metric"], "loss")
        self.assertFalse(read(output / "status.json")["protocol_eligible"])

    def test_weighting_and_separate_stopping_rules(self):
        rows = [{"label": c} for c in CLASSES] + [{"label": "NO_HE"}] * 3
        weights = class_weights(rows, "inverse_sqrt")
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 2, places=6)
        self.assertLess(stopping_key(34, 1.57, "accuracy"), stopping_key(35, 1.61, "accuracy"))
        self.assertGreater(stopping_key(34, 1.57, "loss"), stopping_key(35, 1.61, "loss"))
        for changes in (
            {"weight_decay": float("nan")},
            {"head_lr": 0},
            {"variant": "S3", "feature_scaling": "standard"},
        ):
            with self.assertRaises(ValueError):
                resolve_options(self.args(self.root / "invalid", **changes))


if __name__ == "__main__":
    unittest.main()
