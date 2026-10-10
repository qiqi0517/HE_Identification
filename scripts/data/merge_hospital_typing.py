"""Merge typing data, retaining source uncertainty and user-directed CE2 ground truth."""

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

NAMESPACE = uuid.UUID("2a665fd0-9d69-4d8c-8a8a-390209baf28d")
CLASSES = {
    "NO_HE": "非包虫病（不表示无其他疾病）",
    "CE1": "单囊型",
    "CE2": "多子囊型",
    "CE3": "内囊塌陷型",
    "CE4": "坏死实变型",
    "CE5": "钙化型",
    "AE": "泡型包虫病（细分型未提供）",
}
FOLDERS = {
    "肝包虫病（单囊型）": ("CE1", "source_labeled"),
    "肝包虫病（多子囊型）可能性大": ("CE2", "probable"),
    "肝包虫病(坏死实变型)": ("CE4", "source_labeled"),
    "肝包虫（钙化型）": ("CE5", "source_labeled"),
    "肝泡状棘球蚴病": ("AE", "source_labeled"),
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def image_info(path):
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        return {
            "width": image.width,
            "height": image.height,
            "pixel_sha256": hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
        }


def recover_he_labels(legacy, source):
    """Cross-check retained provenance against original per-image label files."""
    mapping = [
        r
        for r in read_jsonl(legacy / "labels/uuid_mapping.jsonl")
        if r.get("source_dataset") == "HE_DCNN_binary"
    ]
    if not mapping:
        return {}
    original = defaultdict(set)
    for split, folder in (("train", "train_pic"), ("val", "val/pic"), ("test", "test/pic")):
        for line in (source / "data/meta" / f"{split}.txt").read_text().splitlines():
            if line.strip():
                name, raw = line.rsplit(maxsplit=1)
                image = (source / "data" / folder / name).resolve()
                if not image.is_relative_to(source / "data"):
                    raise ValueError("Unsafe HE source path")
                original[sha(image)].add(int(raw))
    for raw in range(5):
        for image in (source / "data/train" / str(raw)).rglob("*"):
            if image.is_file() and image.suffix.lower() in EXTENSIONS:
                original[sha(image)].add(raw)
    result = {}
    for row in mapping:
        raw = row["source_record"]["original_label"]
        if type(raw) is not int or raw not in range(5) or original[row["sha256"]] != {raw}:
            raise ValueError("HE original labels disagree with retained provenance")
        result[row["image"]] = raw
    return result


def merge(legacy, hospital, output, he_source=None, overwrite=False):
    legacy, hospital, output = (Path(p).resolve() for p in (legacy, hospital, output))
    if output.exists() and not overwrite:
        raise FileExistsError("Output already exists; pass --overwrite to replace generated data")
    if output.exists() and not (output / "summary.json").is_file():
        raise ValueError("Refusing to replace a directory without a dataset summary")
    if (
        output.is_relative_to(legacy)
        or output.is_relative_to(hospital)
        or legacy.is_relative_to(output)
        or hospital.is_relative_to(output)
    ):
        raise ValueError("Output must be outside input datasets")
    he_source = Path(
        he_source or Path(__file__).resolve().parents[2] / "data/external/HE_DCNN"
    ).resolve()
    he_labels = recover_he_labels(legacy, he_source)
    records, copies = {}, {}
    old_rows = read_jsonl(legacy / "labels/manifest.jsonl")
    for old in old_rows:
        path = (legacy / old["image"]).resolve()
        if not path.is_relative_to(legacy / "images") or sha(path) != old["sha256"]:
            raise ValueError("Legacy image path/hash mismatch")
        if old["sha256"] in records:
            raise ValueError("Legacy manifest contains duplicate bytes")
        stage = old.get("ce5_label")
        if stage is not None and (type(stage) is not int or stage not in range(5)):
            raise ValueError("Invalid legacy CE stage")
        label = f"CE{stage + 1}" if stage is not None else None
        info = image_info(path)
        if info["pixel_sha256"] != old["pixel_sha256"]:
            raise ValueError("Legacy pixel hash mismatch")
        records[old["sha256"]] = {
            "uuid": old["uuid"],
            "image": old["image"],
            "sha256": old["sha256"],
            **info,
            "legacy_binary_label": old.get("label"),
            "sources": [old["source_dataset"]],
            "typing_evidence": []
            if label is None
            else [
                {
                    "label": label,
                    "status": "source_labeled",
                    "source": old["source_dataset"],
                    "basis": "legacy labels_ce5 / manifest; original source provenance retained",
                }
            ],
            "hospital_source_paths": [],
            "legacy_provenance_image": old["image"],
            "hospital_patient_id": None,
        }
        copies[old["sha256"]] = path
    old_ce5 = {}
    for line in (legacy / "labels/labels_ce5.txt").read_text().splitlines():
        name, index = line.rsplit(maxsplit=1)
        old_ce5[name] = f"CE{int(index) + 1}"
    derived = {
        r["image"]: r["typing_evidence"][0]["label"]
        for r in records.values()
        if r["typing_evidence"]
    }
    if derived != old_ce5:
        raise ValueError("Legacy CE manifest and label list disagree")

    for row in records.values():
        raw = he_labels.get(row["image"])
        row["he_original_label"] = raw
        if "HE_DCNN_binary" in row["sources"] and raw is None:
            raise ValueError("Missing HE original class")
        if raw is not None and row["legacy_binary_label"] != int(raw in (0, 1, 2)):
            raise ValueError("HE binary and original class disagree")
        if row["legacy_binary_label"] == 0:
            row["typing_evidence"].append(
                {
                    "label": "NO_HE",
                    "status": "source_labeled",
                    "source": row["sources"][0],
                    "basis": "User-directed inclusion of all legacy binary negatives; Normal is a source label, not clinical exclusion",
                }
            )
        elif raw == 0:
            row["typing_evidence"].append(
                {
                    "label": "AE",
                    "status": "source_labeled",
                    "source": "HE_DCNN_binary",
                    "basis": "Original meta files + tools/test.py class 0 = AE; SHA256 checked",
                }
            )
        if raw in (1, 2):
            row["coarse_ce_evidence"] = {
                "original_label": raw,
                "repository_display_label": f"CE{raw}",
                "interpretation": "Two CE activity groups; no per-image WHO CE1-CE5 stage established",
                "paper": "https://pubmed.ncbi.nlm.nih.gov/37507196/",
            }

    files, ignored = [], []
    for path in sorted(hospital.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            ignored.append(str(path.relative_to(hospital)))
            continue
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unexpected source file: {path.relative_to(hospital)}")
        rel = path.relative_to(hospital)
        if len(rel.parts) != 2 or rel.parts[0] not in FOLDERS:
            raise ValueError(f"Unknown class directory: {rel}")
        files.append(path)
        label, status = FOLDERS[rel.parts[0]]
        digest, info = sha(path), image_info(path)
        if digest not in records:
            uid = str(uuid.uuid5(NAMESPACE, digest))
            records[digest] = {
                "uuid": uid,
                "image": f"images/{uid}{path.suffix.lower()}",
                "sha256": digest,
                **info,
                "legacy_binary_label": None,
                "sources": [],
                "typing_evidence": [],
                "hospital_source_paths": [],
                "legacy_provenance_image": None,
                "hospital_patient_id": None,
            }
            copies[digest] = path
        row = records[digest]
        if "cooperating_hospital" not in row["sources"]:
            row["sources"].append("cooperating_hospital")
        row["hospital_source_paths"].append(str(rel))
        row["typing_evidence"].append(
            {
                "label": label,
                "status": "user_assigned_ground_truth" if status == "probable" else status,
                "source_status": status,
                "source": "cooperating_hospital",
                "ground_truth_assignment": "User explicitly assigns hospital probable CE2 folder to CE2"
                if status == "probable"
                else None,
                "basis": "provided class directory; not independent clinical adjudication",
                "source_path": str(rel),
                "original_directory_label": rel.parts[0],
            }
        )
    if not files:
        raise ValueError("No hospital images")
    rows = sorted(records.values(), key=lambda row: row["image"])
    pixels = defaultdict(list)
    for row in rows:
        labels = {e["label"] for e in row["typing_evidence"]}
        definite = {
            e["label"]
            for e in row["typing_evidence"]
            if e["status"] in ("source_labeled", "user_assigned_ground_truth")
        }
        conflict = len(labels) > 1 or (bool(labels - {"NO_HE"}) and row["legacy_binary_label"] == 0)
        row["typing_label"] = next(iter(definite)) if len(definite) == 1 and not conflict else None
        row["proposed_typing_label"] = next(iter(labels)) if len(labels) == 1 else None
        row["typing_status"] = (
            "label_conflict"
            if conflict
            else "user_assigned_ground_truth"
            if definite
            and any(e["status"] == "user_assigned_ground_truth" for e in row["typing_evidence"])
            else "source_labeled"
            if definite
            else "probable"
            if labels
            else "missing_typing_label"
        )
        pixels[row["pixel_sha256"]].append(row)
    for group in pixels.values():
        labels = {e["label"] for r in group for e in r["typing_evidence"]}
        if len(labels) > 1 or (
            labels - {"NO_HE"} and any(r["legacy_binary_label"] == 0 for r in group)
        ):
            for row in group:
                row["typing_label"], row["typing_status"] = None, "pixel_label_conflict"
        for row in group:
            row["exact_pixel_group_id"] = "pixels:" + row["pixel_sha256"]
            row["primary_label_eligible"] = row["typing_label"] is not None
    eligible = [r for r in rows if r["primary_label_eligible"]]
    summary = {
        "schema": "he-typing-dataset-v2",
        "legacy_images": len(old_rows),
        "hospital_input_files": len(files),
        "hospital_folder_counts": dict(Counter(p.parent.name for p in files)),
        "audited_unique_images": len(rows),
        "output_images": len(eligible),
        "excluded_images": len(rows) - len(eligible),
        "he_original_label_counts": dict(Counter(str(raw) for raw in he_labels.values())),
        "exact_byte_duplicates_merged": len(old_rows) + len(files) - len(rows),
        "pixel_duplicate_groups": sum(len(g) > 1 for g in pixels.values()),
        "typing_status_counts": dict(Counter(r["typing_status"] for r in rows)),
        "primary_label_counts": dict(Counter(r["typing_label"] for r in eligible)),
        "primary_label_eligible": len(eligible),
        "ignored_metadata_files": ignored,
        "source_roots": {"legacy": str(legacy), "hospital": str(hospital)},
        "split_created": False,
        "image_bytes_unchanged": True,
        "patient_independence_verified": False,
        "clinical_reannotation_performed": False,
        "deidentification_performed": False,
        "near_duplicate_review_performed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".typing-merge-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        labeldir = staging / "labels"
        labeldir.mkdir()
        for row in eligible:
            dest = staging / row["image"]
            shutil.copyfile(copies[row["sha256"]], dest)
            if sha(dest) != row["sha256"]:
                raise ValueError("Copied image hash mismatch")
        # Preserve full legacy patient/source evidence and all previous exclusions.
        shutil.copytree(legacy / "labels", labeldir / "legacy_snapshot")
        write_jsonl(labeldir / "manifest.jsonl", eligible)
        write_jsonl(labeldir / "input_audit.jsonl", rows)
        write_jsonl(
            labeldir / "excluded_records.jsonl",
            [r for r in rows if not r["primary_label_eligible"]],
        )
        (labeldir / "labels_typing.txt").write_text(
            "".join(f"{r['image']} {r['typing_label']}\n" for r in eligible)
        )
        write_json(
            labeldir / "class_mapping.json",
            {
                "schema": "he-typing-labels-v2",
                "classes": CLASSES,
                "prediction_only": {},
                "failure_statuses": ["abstain", "error"],
                "ae_scope": "Disease type only; no AE subtype ground truth",
            },
        )
        write_json(
            labeldir / "pixel_duplicate_groups.json",
            [[r["image"] for r in g] for g in pixels.values() if len(g) > 1],
        )
        write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# 非包虫病与分型数据集 v2\n\n"
            f"原数据 {len(old_rows)} 张 + 医院输入 {len(files)} 张，审核 {len(rows)} 张，仅保留 {len(eligible)} 张有目标标签的图片。\n\n"
            "images/ 保留原始字节及 UUID；旧数据目录和已冻结实验保持不变。\n"
            f"labels/labels_typing.txt 包含 {len(eligible)} 张纳入当前任务真值的图片，标签是 NO_HE / CE1–CE5 / AE 字符串。\n"
            "AE 为泡型疾病类别，不是 CE 分期。不允许输出 CL。医院多子囊型可能性大按用户要求纳入 CE2 真值，保留原始疑似证据。\n"
            "labels/manifest.jsonl 仅包含入选图片、原标签证据、疑似状态、哈希和来源。\n"
            "labels/excluded_records.jsonl 保存冲突和缺少具体分型的排除记录；这些图片不复制到 images/。\n"
            "labels/input_audit.jsonl 保留所有输入的审计记录，原图保留在输入源目录。\n"
            "labels/legacy_snapshot/ 保留旧二分类标签、分期、患者来源映射和排除记录；不可当作新任务主标签。\n"
            "医院 DICOM 风格文件名没有被解释为患者 ID；未进行图内脱敏或临床复标。\n"
            "后续划分前须审核患者/检查关联、图中文字及近重复；当前尚未生成新划分。\n"
        )
        write_json(
            staging / "checksums.json",
            {
                str(p.relative_to(staging)): sha(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        )
        backup = None
        if output.exists():
            backup = (
                Path(tempfile.mkdtemp(prefix=".typing-replace-", dir=output.parent)) / "previous"
            )
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=Path("data/unified"))
    parser.add_argument("--hospital", type=Path, default=Path("data/samples"))
    parser.add_argument("--output", type=Path, default=Path("data/unified_typing_v2"))
    parser.add_argument("--he-source", type=Path, default=Path("data/external/HE_DCNN"))
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace only the generated output dataset"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            merge(args.legacy, args.hospital, args.output, args.he_source, args.overwrite),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
