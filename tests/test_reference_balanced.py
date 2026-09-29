"""验证参考来源/类别配额、完整分组、可复现性及共同测试集。"""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
import reference_balanced as balanced

from reference_split import select_references, validate_split


class ReferenceBalancedTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        provenance = {}
        strata = []
        for source, label, field, value, count in balanced.STRATA:
            names = []
            for _ in range(count + 1):
                name = f"images/{len(rows)}.png"
                names.append(name)
                rows.append(dict(image=name, label=label, sha256=name, source_dataset=source))
                provenance[name] = [{field: value}]
            strata.append(names)
        ids = {r["image"]: r["image"] for r in rows}
        notes = {r["image"]: dict(notes=[], provenance_group_keys=[]) for r in rows}
        # 旧池：Bio四张阳性，HE一个原类别3和三个原类别4阴性。
        old_pool = strata[2] + strata[3] + strata[4][:1] + strata[5]
        refs = [strata[2][0], strata[3][0], strata[4][0], strata[5][0]]
        base = dict(
            version=2,
            seed=42,
            pool_per_class=4,
            reference_pool=old_pool,
            references=refs,
            test=[r["image"] for r in rows if r["image"] not in old_pool],
            samples={
                r["image"]: dict(
                    label=r["label"],
                    sha256=r["sha256"],
                    group_id=ids[r["image"]],
                    **notes[r["image"]],
                )
                for r in rows
            },
        )
        (root / "labels").mkdir()
        (root / "labels/uuid_mapping.jsonl").write_text(
            "".join(
                json.dumps(dict(image=k, source_record=v[0])) + "\n" for k, v in provenance.items()
            )
        )
        path = root / "base.json"
        path.write_text(json.dumps(base))
        return rows, ids, notes, provenance, base, path

    def test_balanced_quotas_group_isolation_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows, ids, notes, provenance, base, path = self.fixture(root)
            before = path.read_bytes()
            with (
                patch.object(balanced, "load_samples", return_value=rows),
                patch.object(balanced, "groups_for", return_value=(ids, notes, [])),
            ):
                new, cohort = balanced.create_balanced_split(root, path)
                repeated, _ = balanced.create_balanced_split(root, path)
            self.assertEqual(new, repeated)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(set(cohort["images"]), set(base["test"]) & set(new["test"]))
            self.assertEqual(len(cohort["images"]), len(base["test"]) - 3)
            lookup = {r["image"]: r for r in rows}
            positives = [lookup[x] for x in new["reference_pool"] if lookup[x]["label"] == 1]
            self.assertEqual(
                Counter(r["source_dataset"] for r in positives),
                {"HE_DCNN_binary": 2, "bioengineering_US_binary": 2},
            )
            negatives = [x for x in new["reference_pool"] if lookup[x]["label"] == 0]
            self.assertEqual(
                Counter(provenance[x][0]["original_label"] for x in negatives), {3: 2, 4: 2}
            )
            for count in (4, 6, 8):
                refs = select_references(new, rows, count)
                self.assertEqual(Counter(r["label"] for r in refs), {0: count // 2, 1: count // 2})
                self.assertEqual(len({r["source_dataset"] for r in refs if r["label"] == 1}), 2)
            validate_split(new, rows)

    def test_unknown_patient_and_shared_groups_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            rows, ids, notes, provenance, base, _ = self.fixture(Path(d))
            he0 = [r["image"] for r in rows if provenance[r["image"]][0].get("original_label") == 0]
            notes[he0[0]]["notes"] = ["unknown patient"]
            selected = balanced.choose_strata(
                rows, ids, notes, provenance, set(base["reference_pool"]), 42
            )
            self.assertEqual(selected[0], [he0[1]])
            notes[he0[0]]["notes"] = []
            ids[he0[0]] = ids[he0[1]] = "shared"
            with self.assertRaisesRegex(ValueError, "Not enough"):
                balanced.choose_strata(
                    rows, ids, notes, provenance, set(base["reference_pool"]), 42
                )

    def test_grouping_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows, ids, notes, _, _, path = self.fixture(root)
            ids[rows[0]["image"]] = "changed"
            with (
                patch.object(balanced, "load_samples", return_value=rows),
                patch.object(balanced, "groups_for", return_value=(ids, notes, [])),
                self.assertRaisesRegex(ValueError, "Grouping changed"),
            ):
                balanced.create_balanced_split(root, path)

    def test_conflicting_provenance_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            rows, ids, notes, provenance, base, _ = self.fixture(Path(d))
            he0 = [r["image"] for r in rows if provenance[r["image"]][0].get("original_label") == 0]
            provenance[he0[0]].append({"original_label": 1})
            selected = balanced.choose_strata(
                rows, ids, notes, provenance, set(base["reference_pool"]), 42
            )
            self.assertEqual(selected[0], [he0[1]])


if __name__ == "__main__":
    unittest.main()
