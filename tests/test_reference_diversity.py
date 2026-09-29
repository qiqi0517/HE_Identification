"""验证来源覆盖对照的固定选择、完整分组和共同测试比较。"""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
from compare_reference_runs import compare
from reference_diversity import choose_additions, create_diverse_split

from reference_split import select_references, validate_split


def fixture(root):
    """八张原池、四张独立 HE 候选和一个不可拆分组。"""
    labels = [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    stages = {0: "CE3", 1: "CE5", 5: "CE5", 7: "CE3"}
    rows, provenance, ids, notes = [], {}, {}, {}
    for i, label in enumerate(labels):
        name = str(i)
        source = "bioengineering_US_binary" if i in stages else "HE_DCNN_binary"
        rows.append({"image": name, "label": label, "sha256": name, "source_dataset": source})
        record = (
            {"label_ce5": stages[i]} if i in stages else {"original_label": i % 2 if label else 4}
        )
        provenance[name] = [record]
        ids[name] = name if i < 12 else "shared"
        notes[name] = {"notes": [] if i < 12 else ["unknown patient"], "provenance_group_keys": []}
    base = {
        "version": 2,
        "seed": 42,
        "pool_per_class": 4,
        "references": ["0", "1", "2", "3"],
        "reference_pool": [str(i) for i in range(8)],
        "test": [str(i) for i in range(8, 14)],
        "samples": {
            r["image"]: {**r, "group_id": ids[r["image"]], **notes[r["image"]]} for r in rows
        },
    }
    (root / "labels").mkdir()
    (root / "labels/uuid_mapping.jsonl").write_text(
        "".join(
            json.dumps({"image": name, "source_record": records[0]}) + "\n"
            for name, records in provenance.items()
        )
    )
    base_path = root / "base.json"
    base_path.write_text(json.dumps(base))
    return rows, provenance, ids, notes, base_path


def build(root):
    rows, provenance, ids, notes, base_path = fixture(root)
    with (
        patch("reference_diversity.load_samples", return_value=rows),
        patch("reference_diversity.groups_for", return_value=(ids, notes, [])),
    ):
        new, cohort = create_diverse_split(root, base_path)
        assert (new, cohort) == create_diverse_split(root, base_path)
    return rows, json.loads(base_path.read_text()), new, cohort


class DiversityTests(unittest.TestCase):
    def test_controlled_changes_and_nested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, old, new, cohort = build(Path(directory))
            lookup = {r["image"]: r for r in rows}
            self.assertEqual(new["references"], old["references"])
            self.assertEqual(set(new["selection_audit"]["removed"]), {"5", "7"})
            self.assertEqual(set(new["test"]) & set(old["test"]), set(cohort["images"]))
            self.assertTrue({"12", "13"} <= set(new["test"]))
            self.assertEqual(
                {x for x in old["reference_pool"] if lookup[x]["label"] == 0},
                {x for x in new["reference_pool"] if lookup[x]["label"] == 0},
            )
            for count in (4, 6, 8):
                refs = select_references(new, rows, count)
                self.assertEqual([r["image"] for r in refs], new["reference_order"][:count])
                self.assertEqual(Counter(r["label"] for r in refs), {0: count // 2, 1: count // 2})
            with self.assertRaises(ValueError):
                select_references(new, rows, 10)

    def test_order_cannot_include_test_or_unbalanced_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, new, _ = build(Path(directory))
            bad = copy.deepcopy(new)
            bad["reference_order"][-1] = new["test"][0]
            with self.assertRaises(ValueError):
                validate_split(bad, rows)
            bad = copy.deepcopy(new)
            # 前六张变为四阴两阳，虽完整池平衡也必须拒绝。
            bad["reference_order"][5], bad["reference_order"][6] = (
                bad["reference_order"][6],
                bad["reference_order"][5],
            )
            with self.assertRaisesRegex(ValueError, "balanced even prefixes"):
                validate_split(bad, rows)
            bad = copy.deepcopy(new)
            bad["reference_order"][0], bad["reference_order"][1] = (
                bad["reference_order"][1],
                bad["reference_order"][0],
            )
            with self.assertRaisesRegex(ValueError, "default four"):
                validate_split(bad, rows)

    def test_no_fallback_to_uncertain_or_multi_image_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, provenance, ids, notes, base_path = fixture(Path(directory))
            pool = set(json.loads(base_path.read_text())["reference_pool"])
            for i in (8, 10):
                notes[str(i)]["notes"] = ["unknown"]
            with self.assertRaisesRegex(ValueError, "original label 0"):
                choose_additions(rows, ids, notes, provenance, pool, 42)
            notes["8"]["notes"] = []
            ids["8"] = ids["10"] = "same_patient"
            with self.assertRaises(ValueError):
                choose_additions(rows, ids, notes, provenance, pool, 42)

    def test_conflicting_raw_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, provenance, ids, notes, base_path = fixture(Path(directory))
            for i in (8, 10):
                provenance[str(i)].append({"original_label": 1})
            with self.assertRaises(ValueError):
                choose_additions(
                    rows,
                    ids,
                    notes,
                    provenance,
                    set(json.loads(base_path.read_text())["reference_pool"]),
                    42,
                )


class ComparisonTests(unittest.TestCase):
    def setup_runs(self, root):
        rows, old, new, cohort = build(root)
        lookup = {r["image"]: r for r in rows}
        cohort_path = root / "cohort.json"
        cohort_path.write_text(json.dumps(cohort))
        for name, split in (("old", old), ("new", new)):
            path = root / name
            path.mkdir()
            split_bytes = json.dumps(split).encode()
            (path / "split.json").write_bytes(split_bytes)
            (path / "prompt.txt").write_text("same prompt")
            config = {
                "dry_run": False,
                "selected": len(split["test"]),
                "reference_count": 8,
                "model_config": {"provider": "mock", "model": "offline"},
                "mode": "reference",
                "prompt_sha256": hashlib.sha256(b"same prompt").hexdigest(),
                "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "labels_sha256": "same-labels",
                "seed": 42,
                "decision_threshold": 0.5,
            }
            (path / "run.json").write_text(json.dumps(config))
            (path / "samples.json").write_text(json.dumps([lookup[x] for x in split["test"]]))
            (path / "references.json").write_text(json.dumps(select_references(split, rows, 8)))
            predictions = []
            for image in split["test"]:
                # 池变更造成的病例进出具有极端差异；共同病例仍应得到零差值。
                probability = (
                    0.6 if image in cohort["images"] else (0.01 if name == "old" else 0.99)
                )
                predictions.append(
                    {
                        "image": image,
                        "true_label": lookup[image]["label"],
                        "source_dataset": lookup[image]["source_dataset"],
                        "p_he": probability,
                        "status": "ok",
                    }
                )
            (path / "predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in predictions)
            )
        cohort["parent_split_sha256"] = hashlib.sha256(
            (root / "old/split.json").read_bytes()
        ).hexdigest()
        cohort_path.write_text(json.dumps(cohort))
        return cohort_path

    def test_excludes_moved_samples_not_just_current_test_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.setup_runs(root)
            result = compare(root / "old", root / "new", path)
            self.assertEqual(result["planned_common"], 4)
            self.assertEqual(result["delta_log_loss_new_minus_old"], 0)
            self.assertEqual(result["delta_brier_new_minus_old"], 0)

    def test_rejects_cherry_picked_cohort_or_changed_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.setup_runs(root)
            cohort = json.loads(path.read_text())
            cohort["images"].pop()
            path.write_text(json.dumps(cohort))
            with self.assertRaisesRegex(ValueError, "complete old/new test intersection"):
                compare(root / "old", root / "new", path)
            (root / "new/prompt.txt").write_text("different prompt")
            with self.assertRaisesRegex(ValueError, "Prompt snapshot"):
                compare(root / "old", root / "new", path)

    def test_incomplete_run_rejected_and_abstention_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = self.setup_runs(root)
            path = root / "new/predictions.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                compare(root / "old", root / "new", cohort_path)
            shared = set(json.loads(cohort_path.read_text())["images"])
            next(r for r in rows if r["image"] in shared).update(status="abstain", p_he=None)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = compare(root / "old", root / "new", cohort_path)
            self.assertEqual(result["common_valid"], 3)
            self.assertEqual(result["coverage_on_frozen_cohort"]["new"]["abstentions"], 1)


if __name__ == "__main__":
    unittest.main()
