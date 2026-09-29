"""检查真实权重、疾病概率边界以及参考/测试隔离。"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from he_checkpoint import binary_probability, checkpoint_status, load_he_candidate, map_mmcls_keys
from run_experiment import validated_reference_pool
from ultrasound_encoder import UltrasoundEncoder, nearest_references


class ExpertTests(unittest.TestCase):
    def test_missing_weights_abstains(self):
        result = checkpoint_status()
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["p_he"])

    def test_imagenet_head_rejected(self):
        with patch(
            "he_checkpoint.safe_state_dict", return_value={"fc.weight": torch.zeros(1000, 2048)}
        ):
            with self.assertRaises(ValueError):
                load_he_candidate(Path("unused.pth"))

    def test_incomplete_he_checkpoint_rejected(self):
        with patch(
            "he_checkpoint.safe_state_dict", return_value={"head.fc.weight": torch.zeros(5, 2048)}
        ):
            with self.assertRaises(RuntimeError):
                load_he_candidate(Path("unused.pth"))

    def test_unlabeled_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "labels").mkdir()
            (root / "labels/labels.txt").write_text("other 1\n")
            path = root / "split.json"
            path.write_text(
                json.dumps({"reference_pool": ["a"], "test": ["b"], "references": ["a"]})
            )
            with self.assertRaises((ValueError, KeyError)):
                validated_reference_pool(root, path)

    def test_probability_sums_three_positive_classes(self):
        result = binary_probability(torch.zeros(2, 5), (0, 1, 2), class_mapping_confirmed=True)
        torch.testing.assert_close(result, torch.tensor([0.6, 0.6]))

    def test_unconfirmed_mapping_rejected(self):
        with self.assertRaises(ValueError):
            binary_probability(torch.zeros(1, 5), (0, 1, 2), class_mapping_confirmed=False)

    def test_invalid_logits_rejected(self):
        for logits in [torch.zeros(1, 1000), torch.full((1, 5), float("nan"))]:
            with self.assertRaises(ValueError):
                binary_probability(logits, (0,), class_mapping_confirmed=True)

    def test_invalid_class_sets_rejected(self):
        for indices in [(), (0, 0), (-1,), (True,), (0, 1, 2, 3, 4)]:
            with self.assertRaises(ValueError):
                binary_probability(torch.zeros(1, 5), indices, class_mapping_confirmed=True)

    def test_mmcls_mapping_and_collision(self):
        state = map_mmcls_keys(
            {"module.backbone.conv1.weight": torch.zeros(1), "head.fc.bias": torch.zeros(5)}
        )
        self.assertEqual(set(state), {"conv1.weight", "fc.bias"})
        with self.assertRaises(ValueError):
            map_mmcls_keys(
                {"backbone.conv1.weight": torch.zeros(1), "conv1.weight": torch.zeros(1)}
            )

    def test_retrieval_excludes_self_and_has_deterministic_ties(self):
        features = np.zeros((3, 192), dtype=np.float32)
        features[:, 0] = 1
        results = nearest_references(features[0], features, ["q", "z", "a"], "q", 4)
        self.assertEqual([r["image"] for r in results], ["a", "z"])

    def test_split_leak_rejected_before_reading_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            path.write_text(
                json.dumps({"reference_pool": ["a"], "test": ["a"], "references": ["a"]})
            )
            with self.assertRaises(ValueError):
                validated_reference_pool(Path(temp), path)

    def test_checkpoint_fingerprint_rejects_unknown_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pth"
            path.write_bytes(b"not trusted")
            with self.assertRaises(ValueError):
                UltrasoundEncoder(path)

    @unittest.skipUnless((ROOT / "weights/TinyUSFM.pth").is_file(), "Public weights not downloaded")
    def test_real_checkpoint_matches_author_forward_features(self):
        # 作者源码快照只用于独立一致性测试，不参与日常工具运行。
        spec = importlib.util.spec_from_file_location(
            "author_tiny", ROOT / "research/tiny_model_tinyusfm.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.set_num_threads(2)
        ours = UltrasoundEncoder(ROOT / "weights/TinyUSFM.pth")
        author = module.TinyUSFM(num_classes=0, global_pool=False).eval()
        state = torch.load(ROOT / "weights/TinyUSFM.pth", weights_only=True, map_location="cpu")
        author.load_state_dict(state, strict=True)
        torch.manual_seed(42)
        sample = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            expected = author.model.forward_features(sample)
            actual = ours.model(sample)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(tuple(actual.shape), (1, 192))


if __name__ == "__main__":
    unittest.main()
