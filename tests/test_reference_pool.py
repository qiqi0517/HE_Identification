"""小型平衡池、完整患者分组和可扩展示例的回归检查。"""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
from prepare_reference_split import balanced_groups, create_split

from reference_split import select_references, validate_split


class BalancedPoolTests(unittest.TestCase):
    def fixture(self):
        rows = [
            {"image": str(i), "label": i % 2, "sha256": str(i), "source_dataset": "fixture"}
            for i in range(12)
        ]
        split = {
            "version": 2,
            "pool_per_class": 4,
            "seed": 42,
            "reference_pool": [str(i) for i in range(8)],
            "references": ["1", "3", "0", "2"],
            "test": [str(i) for i in range(8, 12)],
            "samples": {r["image"]: {**r, "group_id": r["image"]} for r in rows},
        }
        return rows, split

    def test_expand_preserves_order_balance_and_test_set(self):
        rows, split = self.fixture()
        test_before = validate_split(split, rows)[0]
        prior = []
        for count in (4, 6, 8):
            refs = select_references(split, rows, count)
            self.assertEqual(Counter(r["label"] for r in refs), {0: count // 2, 1: count // 2})
            self.assertEqual([r["image"] for r in refs[:4]], split["references"])
            self.assertEqual(refs[: len(prior)], prior)
            self.assertEqual(len({r["image"] for r in refs}), count)
            self.assertEqual(refs, select_references(split, list(reversed(rows)), count))
            self.assertEqual(validate_split(split, rows)[0], test_before)
            prior = refs
        self.assertEqual(select_references(split, rows), prior[:4])

    def test_invalid_or_unavailable_count_rejected(self):
        rows, split = self.fixture()
        for count in (0, 2, 5, -4, True, 6.0, 10):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_references(split, rows, count)

    def test_unbalanced_v2_rejected_but_legacy_supported(self):
        rows, split = self.fixture()
        split["reference_pool"].remove("6")
        split["test"].append("6")
        with self.assertRaisesRegex(ValueError, "balanced class quota"):
            validate_split(split, rows)
        split["version"] = 1
        self.assertEqual(len(select_references(split, rows)), 4)
        with self.assertRaisesRegex(ValueError, "Insufficient balanced"):
            select_references(split, rows, 8)

    def test_group_selection_avoids_greedy_dead_end(self):
        groups = {
            "mixed": [{"label": 0}, {"label": 1}],
            "negative_pair": [{"label": 0}, {"label": 0}],
            "positive_pair": [{"label": 1}, {"label": 1}],
        }
        self.assertEqual(
            balanced_groups(groups, list(groups), set(), 2),
            {"negative_pair", "positive_pair"},
        )
        with self.assertRaisesRegex(ValueError, "without splitting"):
            balanced_groups(groups, list(groups), {"mixed"}, 2)

    def test_mandatory_ineligible_or_oversized_group_rejected(self):
        groups = {"large": [{"label": 0}] * 3, "positive": [{"label": 1}] * 2}
        with self.assertRaisesRegex(ValueError, "ineligible"):
            balanced_groups(groups, ["positive"], {"large"}, 2)
        with self.assertRaisesRegex(ValueError, "exceed"):
            balanced_groups(groups, list(groups), {"large"}, 2)

    def test_create_split_migration_and_new_split(self):
        rows, previous = self.fixture()
        ids = {r["image"]: r["image"] for r in rows}
        notes = {r["image"]: {"notes": [], "provenance_group_keys": []} for r in rows}
        # 关联信息不足的整组即使有标签，也不能被填充到参考池。
        notes["10"]["notes"] = ["unknown patient"]
        notes["11"]["notes"] = ["unknown patient"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels").mkdir()
            (root / "labels/uuid_mapping.jsonl").write_text(
                "".join(json.dumps({"image": r["image"], "source_record": {}}) + "\n" for r in rows)
            )
            parent = root / "old.json"
            parent.write_text(json.dumps(previous))
            with (
                patch("prepare_reference_split.load_samples", return_value=rows),
                patch("prepare_reference_split.groups_for", return_value=(ids, notes, [])),
            ):
                migrated = create_split(root, base_split=parent)
                repeat = create_split(root, base_split=parent)
                new = create_split(root)
            self.assertEqual(migrated, repeat)
            self.assertEqual(migrated["references"], previous["references"])
            self.assertEqual(set(migrated["reference_pool"]), set(previous["reference_pool"]))
            self.assertIn("parent_split_sha256", migrated)
            for result in (migrated, new):
                self.assertEqual(result["counts"]["reference_pool"]["labels"], {0: 4, 1: 4})
                self.assertTrue({"10", "11"} <= set(result["test"]))
                validate_split(result, rows)


if __name__ == "__main__":
    unittest.main()
