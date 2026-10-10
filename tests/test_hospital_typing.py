"""Preserve source uncertainty under explicit CE2 assignment; reject conflicts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "merge_hospital_typing",
    Path(__file__).resolve().parents[1] / "scripts/data/merge_hospital_typing.py",
)
merge_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_module)


class HospitalTypingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy"
        self.hospital = self.root / "hospital"
        self.output = self.root / "merged"
        (self.legacy / "images").mkdir(parents=True)
        (self.legacy / "labels").mkdir()
        self.hospital.mkdir()
        self.old = self.legacy / "images/old.png"
        Image.new("RGB", (12, 9), "red").save(self.old)
        self.row = {
            "uuid": "old",
            "image": "images/old.png",
            "sha256": merge_module.sha(self.old),
            **merge_module.image_info(self.old),
            "label": 1,
            "ce5_label": 2,
            "source_dataset": "fixture",
        }
        merge_module.write_jsonl(self.legacy / "labels/manifest.jsonl", [self.row])
        (self.legacy / "labels/labels_ce5.txt").write_text("images/old.png 2\n")
        (self.legacy / "labels/uuid_mapping.jsonl").write_text('{"patient_group":"known"}\n')

    def sample(self, folder, filename="sample.JPG", color="blue", content=None):
        path = self.hospital / folder / filename
        path.parent.mkdir(exist_ok=True)
        if content is None:
            Image.new("RGB", (10, 11), color).save(path)
        else:
            path.write_bytes(content)
        return path

    def test_user_assigned_ce2_preserves_probable_evidence_and_legacy(self):
        before = self.old.read_bytes()
        self.sample("肝包虫病（多子囊型）可能性大")
        definite = self.sample("肝包虫病（单囊型）", color="green")
        result = merge_module.merge(self.legacy, self.hospital, self.output)
        self.assertEqual(result["output_images"], 3)
        self.assertEqual(result["audited_unique_images"], 3)
        self.assertEqual(result["primary_label_counts"], {"CE1": 1, "CE2": 1, "CE3": 1})
        self.assertEqual(merge_module.read_jsonl(self.output / "labels/excluded_records.jsonl"), [])
        ce2 = next(
            r
            for r in merge_module.read_jsonl(self.output / "labels/manifest.jsonl")
            if r["typing_label"] == "CE2"
        )
        self.assertTrue(ce2["primary_label_eligible"])
        self.assertEqual(ce2["typing_status"], "user_assigned_ground_truth")
        self.assertEqual(ce2["typing_evidence"][0]["source_status"], "probable")
        self.assertTrue((self.output / ce2["image"]).exists())
        self.assertEqual(self.old.read_bytes(), before)
        self.assertEqual((self.output / "images/old.png").read_bytes(), before)
        rows = merge_module.read_jsonl(self.output / "labels/manifest.jsonl")
        ce1 = next(r for r in rows if r["typing_label"] == "CE1")
        self.assertEqual((self.output / ce1["image"]).read_bytes(), definite.read_bytes())
        self.assertEqual(
            (self.output / "labels/legacy_snapshot/uuid_mapping.jsonl").read_bytes(),
            (self.legacy / "labels/uuid_mapping.jsonl").read_bytes(),
        )
        with self.assertRaises(FileExistsError):
            merge_module.merge(self.legacy, self.hospital, self.output)

    def test_duplicate_conflicting_folders_quarantined(self):
        p = self.sample("肝包虫病（单囊型）")
        self.sample("肝包虫（钙化型）", content=p.read_bytes())
        result = merge_module.merge(self.legacy, self.hospital, self.output)
        self.assertEqual(result["exact_byte_duplicates_merged"], 1)
        self.assertEqual(result["primary_label_eligible"], 1)
        review = merge_module.read_jsonl(self.output / "labels/excluded_records.jsonl")
        self.assertEqual(len(review[0]["typing_evidence"]), 2)
        self.assertIsNone(review[0]["typing_label"])
        self.assertFalse((self.output / review[0]["image"]).exists())

    def test_overwrite_keeps_previous_output_when_new_inputs_fail(self):
        self.sample("肝包虫病（多子囊型）可能性大")
        merge_module.merge(self.legacy, self.hospital, self.output)
        before = (self.output / "labels/manifest.jsonl").read_bytes()
        original = self.old.read_bytes()
        merge_module.merge(self.legacy, self.hospital, self.output, overwrite=True)
        self.assertEqual((self.output / "labels/manifest.jsonl").read_bytes(), before)
        self.sample("unknown-class")
        with self.assertRaises(ValueError):
            merge_module.merge(self.legacy, self.hospital, self.output, overwrite=True)
        self.assertEqual((self.output / "labels/manifest.jsonl").read_bytes(), before)
        self.assertEqual(self.old.read_bytes(), original)

    def test_recover_he_ae_negatives_and_exclude_coarse_ce(self):
        self.sample("肝包虫病（单囊型）")
        source = self.root / "he_source"
        (source / "data/meta").mkdir(parents=True)
        (source / "data/train_pic").mkdir()
        (source / "data/train_pic/old.png").write_bytes(self.old.read_bytes())
        for split in ("val", "test"):
            (source / "data/meta" / f"{split}.txt").write_text("")
        self.row["source_dataset"] = "HE_DCNN_binary"
        self.row["ce5_label"] = None
        (self.legacy / "labels/labels_ce5.txt").write_text("")
        for raw, expected in ((0, "AE"), (1, None), (2, None), (3, "NO_HE"), (4, "NO_HE")):
            with self.subTest(original_label=raw):
                self.row["label"] = int(raw < 3)
                merge_module.write_jsonl(self.legacy / "labels/manifest.jsonl", [self.row])
                merge_module.write_jsonl(
                    self.legacy / "labels/uuid_mapping.jsonl",
                    [
                        {
                            "image": self.row["image"],
                            "sha256": self.row["sha256"],
                            "source_dataset": "HE_DCNN_binary",
                            "source_record": {"original_label": raw},
                        }
                    ],
                )
                (source / "data/meta/train.txt").write_text(f"old.png {raw}\n")
                out = self.root / f"case{raw}"
                merge_module.merge(self.legacy, self.hospital, out, source)
                rows = merge_module.read_jsonl(out / "labels/manifest.jsonl")
                old = [r for r in rows if r["image"] == self.row["image"]]
                if expected:
                    self.assertEqual(old[0]["typing_label"], expected)
                else:
                    self.assertFalse(old)
                    self.assertFalse((out / self.row["image"]).exists())
        (source / "data/meta/train.txt").write_text("old.png 0\n")
        with self.assertRaises(ValueError):
            merge_module.merge(self.legacy, self.hospital, self.output, source)

    def test_unknown_directory_fails_without_publishing(self):
        self.sample("unknown-class")
        with self.assertRaises(ValueError):
            merge_module.merge(self.legacy, self.hospital, self.output)
        self.assertFalse(self.output.exists())

    def test_legacy_label_mismatch_fails(self):
        self.sample("肝包虫病（单囊型）")
        (self.legacy / "labels/labels_ce5.txt").write_text("images/old.png 0\n")
        with self.assertRaises(ValueError):
            merge_module.merge(self.legacy, self.hospital, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
