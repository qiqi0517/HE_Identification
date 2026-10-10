"""Immutable metadata snapshots and role-specific image access."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from reference_split import validate_typing_split

from . import CLASSES, SCHEMA

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_lines(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


def code_hash():
    paths = sorted(Path(__file__).parent.glob("*.py")) + [
        PROJECT / "src/reference_split.py",
        PROJECT / "experiments/expert_lab/src/ultrasound_encoder.py",
    ]
    return signature({str(p.relative_to(PROJECT)): sha(p) for p in paths})


def metadata(dataset):
    """Read labels/metadata only: preparation/training never decode test images."""
    dataset = Path(dataset).resolve()
    manifest_path = dataset / "labels/manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    lookup = {r["image"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate manifest image")
    samples = {}
    label_path = dataset / "labels/labels_typing.txt"
    for line in label_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, label = line.rsplit(maxsplit=1)
        row = lookup[name]
        image_path(dataset, name)
        if name in samples or label not in CLASSES:
            raise ValueError("Duplicate/invalid label")
        if row["typing_label"] != label or row.get("primary_label_eligible") is not True:
            raise ValueError("Excluded or inconsistent sample")
        samples[name] = dict(
            image=name,
            label=label,
            sha256=row["sha256"],
            source_dataset="+".join(sorted(row["sources"])),
        )
    if not samples or set(samples) != set(lookup):
        raise ValueError("Manifest/label coverage differs")
    return samples, {"labels": sha(label_path), "manifest": sha(manifest_path)}


def image_path(dataset, name):
    root = Path(dataset).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root / "images"):
        raise ValueError("Image path outside dataset/images")
    return path


def prepare(dataset, source_split, output, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    samples, hashes = metadata(dataset)
    source = read(source_split)
    if source.get("dataset_labels_sha256") != hashes["labels"]:
        raise ValueError("Source split label hash changed")
    validate_typing_split(source, list(samples.values()))
    p = source["partitions"]
    mapped = dict(train=p["reference"] + p["search"], validation=p["selection"], test=p["test"])
    result = dict(
        schema=SCHEMA,
        classes=list(CLASSES),
        dataset=str(Path(dataset).resolve()),
        dataset_hashes=hashes,
        source_split_sha256=sha(source_split),
        source_split=source,
        partitions=mapped,
        samples=source["samples"],
        image_version="unified_typing_v2_original_exploratory",
        limitations=source.get("limitations", []),
        audit_status="patient/text/clinical/pretraining_overlap_review_pending",
    )
    result["counts"] = {
        role: dict(total=len(names), labels=dict(Counter(samples[n]["label"] for n in names)))
        for role, names in mapped.items()
    }
    validate_mapping(result, dataset)
    write(output, result)
    return result["counts"]


def validate_mapping(split, dataset):
    if split.get("schema") != SCHEMA or split.get("classes") != list(CLASSES):
        raise ValueError("Incompatible expert split/class order")
    current, hashes = metadata(dataset)
    if hashes != split["dataset_hashes"]:
        raise ValueError("Stale dataset labels/manifest")
    source = split["source_split"]
    validate_typing_split(source, list(current.values()))
    if split["samples"] != source["samples"]:
        raise ValueError("Modified source sample metadata")
    p = source["partitions"]
    expected = dict(train=p["reference"] + p["search"], validation=p["selection"], test=p["test"])
    if split["partitions"] != expected:
        raise ValueError("Modified expert role mapping")
    group_roles, bytes_roles = {}, {}
    for role, names in expected.items():
        for name in names:
            row = split["samples"][name]
            for key, seen in [(row["group_id"], group_roles), (row["sha256"], bytes_roles)]:
                if seen.get(key, role) != role:
                    raise ValueError("Group/exact duplicate leakage")
                seen[key] = role
    return split


def load_split(path, dataset=None):
    split = read(path)
    root = Path(dataset or split["dataset"]).resolve()
    return validate_mapping(split, root), root


def rows_for(split, role):
    return [dict(image=n, **split["samples"][n]) for n in split["partitions"][role]]


def verify_images(dataset, rows):
    for row in rows:
        if sha(image_path(dataset, row["image"])) != row["sha256"]:
            raise ValueError(f"Image changed: {row['image']}")
