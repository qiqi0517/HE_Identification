"""Check grouped allocation, leakage rejection and immutable split output."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
import prepare_auto_harness_split as split


class AutoHarnessSplitTests(unittest.TestCase):
    def fixture(self):
        samples, groups, audit = [], {}, {}
        strata = [
            ("bioengineering_US_binary", 1, 80),
            ("HE_DCNN_binary", 1, 22),
            ("web_liver_ultrasound_20260918", 1, 3),
            ("HE_DCNN_binary", 0, 14),
            ("web_liver_ultrasound_20260918", 0, 100),
        ]
        for source, label, count in strata:
            for i in range(count):
                name = f"{source}/{label}/{i}"
                gid = name
                if source == "HE_DCNN_binary" and i < 8:
                    gid = "mixed_unknown"
                elif source.startswith("web"):
                    if label:
                        gid = "publication"
                    elif i < 56:
                        gid = "normal_component"
                groups.setdefault(gid, []).append(name)
                samples.append(dict(image=name, source_dataset=source, label=label))
                audit[name] = dict(group_id=gid, flags=[])
        return samples, groups, audit

    def test_quotas_group_integrity_and_determinism(self):
        samples, groups, audit = self.fixture()
        parts = split.assign(groups, samples, audit, 42)
        self.assertEqual(parts, split.assign(groups, samples, audit, 42))
        split.validate(parts, samples, audit, [])
        self.assertEqual(
            {k: len(v) for k, v in parts.items()},
            dict(reference=8, search=111, selection=40, test=60),
        )
        lookup = {r["image"]: r for r in samples}
        self.assertEqual(Counter(lookup[n]["label"] for n in parts["test"]), {1: 28, 0: 32})
        self.assertTrue(set(groups["mixed_unknown"]) <= set(parts["search"]))
        self.assertTrue(set(groups["normal_component"]) <= set(parts["search"]))

    def test_rejects_group_leakage(self):
        samples, groups, audit = self.fixture()
        parts = split.assign(groups, samples, audit, 42)
        name = groups["normal_component"][0]
        parts["search"].remove(name)
        parts["selection"].append(name)
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            split.validate(parts, samples, audit, [])

    def test_infeasible_quota_fails_without_splitting(self):
        with self.assertRaises(ValueError):
            split.pick_subset(["big"], {"big": ["a"] * 100}, 28, set())

    def test_reserve_reference_singletons(self):
        groups = {"a": ["a"], "b": ["b"], "c": ["c"], "pair": ["d", "e"]}
        result = split.pick_subset(list(groups), groups, 3, {"a", "b", "c"}, reserve=2)
        self.assertIn("pair", result)
        self.assertEqual(len(result & {"a", "b", "c"}), 1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                split.write_bundle(Path(root), {}, {})


if __name__ == "__main__":
    unittest.main()
