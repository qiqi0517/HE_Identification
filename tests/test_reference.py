"""验证固定划分的完整性、分组隔离和参考图请求顺序。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from he_agent import build_request
from models import ImagePart, TextPart
from reference_split import validate_split


class ReferenceTests(unittest.TestCase):
    def fixture(self):
        rows = [{"image": str(i), "label": i % 2, "sha256": str(i)} for i in range(6)]
        split = {
            "reference_pool": ["0", "1", "2", "3"],
            "references": ["0", "1", "2", "3"],
            "test": ["4", "5"],
            "samples": {r["image"]: {**r, "group_id": r["image"]} for r in rows},
        }
        return rows, split

    def test_disjoint_fixed_split(self):
        rows, s = self.fixture()
        test, refs = validate_split(s, rows)
        self.assertEqual([r["image"] for r in test], ["4", "5"])
        self.assertEqual(len(refs), 4)

    def test_group_leakage_rejected(self):
        rows, s = self.fixture()
        s["samples"]["4"]["group_id"] = "0"
        with self.assertRaises(ValueError):
            validate_split(s, rows)

    def test_test_reference_overlap_rejected(self):
        rows, s = self.fixture()
        s["references"][0] = "4"
        with self.assertRaises(ValueError):
            validate_split(s, rows)

    def test_stale_labels_rejected(self):
        rows, s = self.fixture()
        rows[4]["label"] = 1
        with self.assertRaises(ValueError):
            validate_split(s, rows)

    def test_payload_four_refs_one_test(self):
        refs = [{"bytes": b"reference", "mime": "image/png", "label": i % 2} for i in range(4)]
        request = build_request(b"test", "image/png", "prompt", refs)
        blocks = request.content
        self.assertEqual(sum(isinstance(x, ImagePart) for x in blocks), 5)
        self.assertEqual(sum(isinstance(x, TextPart) and "诊断标签" in x.text for x in blocks), 4)
        self.assertEqual(blocks[-1].data, b"test")
        self.assertFalse(hasattr(request, "true_label"))


if __name__ == "__main__":
    unittest.main()
