"""Generate an audited, immutable internal split; never read predictions or call models.

Normal records lack patient IDs. Their source-level split group is relaxed only with
an explicit policy flag; recorded patients, publication groups and dHash edges remain
indivisible. This is an internal image/group split, not patient-independent validation.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from he_agent import load_samples  # noqa: E402

ROLES = ("reference", "search", "selection", "test")
STRATA = ("bio_positive", "he_positive", "web_positive", "he_negative", "normal_negative")
# Publication group of three web positives is kept together; Bio compensates.
TEST_QUOTA = dict(zip(STRATA, (19, 6, 3, 4, 28)))
REFERENCE_QUOTA = dict(zip(STRATA, (2, 2, 0, 2, 2)))
# The conservative Normal dHash component (56 images) is retained in search.
# All eight remaining HE negatives belong to the mixed unknown-patient group.
SELECTION_QUOTA = dict(zip(STRATA, (22, 4, 0, 0, 14)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def stratum(row):
    source, label = row["source_dataset"], row["label"]
    if source == "bioengineering_US_binary" and label == 1:
        return "bio_positive"
    if source == "HE_DCNN_binary":
        return "he_positive" if label else "he_negative"
    if source == "web_liver_ultrasound_20260918":
        return "web_positive" if label else "normal_negative"
    raise ValueError(f"Unexpected source/label: {source}/{label}")


def build_groups(dataset, samples, allow_normal_image_groups=False):
    records = defaultdict(list)
    for line in (dataset / "labels/uuid_mapping.jsonl").read_text().splitlines():
        row = json.loads(line)
        records[row["image"]].append(row["source_record"])
    parent = {s["image"]: s["image"] for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[max(find(a), find(b))] = min(find(a), find(b))

    seen, dhashes, audit = {}, {}, {}
    for s in samples:
        name, tokens, flags = s["image"], [], []
        source = s["source_dataset"]
        if not records[name]:
            raise ValueError("Missing provenance")
        for r in records[name]:
            raw_label = r.get("label")
            if raw_label is not None and raw_label != s["label"]:
                raise ValueError("Provenance label conflict")
            if r.get("modality") not in (None, "US") or r.get("organ") not in (None, "liver"):
                raise ValueError("Unexpected modality/organ")
            if source == "HE_DCNN_binary":
                keys = r.get("patient_groups", [])
                tokens.extend("he_patient:" + Path(k).name for k in keys)
                if not keys:
                    tokens.append("he_unknown_patients")
                    flags.append("unknown_patient_conservatively_grouped")
            elif source == "bioengineering_US_binary":
                if not r.get("patient_key"):
                    raise ValueError("Missing Bio patient key")
                tokens.append("bio_patient:" + r["patient_key"])
                flags.append("source_local_patient_identity_only")
            elif r.get("source_id") == "zenodo_7272660":
                if not allow_normal_image_groups:
                    tokens.append("normal_source:zenodo_7272660")
                if r.get("patient_id"):
                    tokens.append("normal_patient:" + str(r["patient_id"]))
                flags.extend(("patient_unknown", "normal_source_label_not_he_exclusion"))
            else:
                tokens.append("web_publication:" + r["split_group"])
                flags.append("publication_group_not_confirmed_patient")
        with Image.open(dataset / name) as im:
            rgb = im.convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            gray = im.convert("L").resize((9, 8), Image.Resampling.BICUBIC)
            v = list(gray.getdata())
            dhashes[name] = sum(
                int(v[y * 9 + x] > v[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8)
                for x in range(8)
            )
        tokens.extend(("bytes:" + s["sha256"], "pixels:" + pixel_hash))
        for token in tokens:
            if token in seen:
                union(name, seen[token])
            seen[token] = name
        audit[name] = {
            "group_keys": sorted(set(tokens)),
            "flags": sorted(set(flags)),
            "dhash": f"{dhashes[name]:016x}",
            "computed_pixel_sha256": pixel_hash,
            "patient_independence_verified": False,
        }
    near = []
    names = sorted(parent)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            distance = (dhashes[a] ^ dhashes[b]).bit_count()
            if distance <= 4:
                union(a, b)
                near.append(
                    {
                        "a": a,
                        "b": b,
                        "distance": distance,
                        "decision": "keep_together_conservative_candidate_not_proven_duplicate",
                    }
                )
    members = defaultdict(list)
    for name in names:
        members[find(name)].append(name)
    groups = {}
    for names in members.values():
        gid = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
        groups[gid] = names
        for name in names:
            audit[name]["group_id"] = gid
    return groups, audit, near


def pick_subset(candidates, groups, target, eligible, reserve=0):
    """Exact image count without splitting groups; reserve eligible reference singletons."""
    available_refs = sum(g in eligible for g in candidates)
    states = {(0, 0): ()}
    for gid in candidates:
        size, ref = len(groups[gid]), int(gid in eligible)
        for (count, used_refs), selected in list(states.items()):
            key = (count + size, used_refs + ref)
            if key[0] <= target and available_refs - key[1] >= reserve:
                states.setdefault(key, (*selected, gid))
    matches = [value for (count, _), value in states.items() if count == target]
    if not matches:
        raise ValueError(f"Cannot satisfy grouped quota {target} with reference reserve {reserve}")
    return set(matches[0])


def assign(groups, samples, audit, seed):
    lookup = {s["image"]: s for s in samples}
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    pure = defaultdict(list)
    mixed = []
    for gid in order:
        strata = {stratum(lookup[n]) for n in groups[gid]}
        if len(strata) == 1:
            pure[next(iter(strata))].append(gid)
        else:
            mixed.append(gid)
    eligible = {
        gid
        for gid in order
        if len(groups[gid]) == 1
        and "unknown_patient_conservatively_grouped" not in audit[groups[gid][0]]["flags"]
    }
    role = dict.fromkeys(mixed, "search")
    for key in STRATA:
        candidates = pure[key]
        test = pick_subset(candidates, groups, TEST_QUOTA[key], eligible, REFERENCE_QUOTA[key])
        refs = [g for g in candidates if g not in test and g in eligible][: REFERENCE_QUOTA[key]]
        if len(refs) != REFERENCE_QUOTA[key]:
            raise ValueError("Reference quota unavailable")
        remaining = [g for g in candidates if g not in test and g not in refs]
        selection = pick_subset(remaining, groups, SELECTION_QUOTA[key], set())
        for g in candidates:
            role[g] = (
                "test"
                if g in test
                else "reference"
                if g in refs
                else ("selection" if g in selection else "search")
            )
    partitions = {r: sorted(n for g in order if role[g] == r for n in groups[g]) for r in ROLES}
    refs_by_label = {
        y: [n for n in partitions["reference"] if lookup[n]["label"] == y] for y in (0, 1)
    }
    partitions["reference"] = [n for pair in zip(refs_by_label[1], refs_by_label[0]) for n in pair]
    return partitions


def validate(partitions, samples, audit, near):
    lookup = {r["image"]: r for r in samples}
    names = [n for role in ROLES for n in partitions[role]]
    if len(names) != len(set(names)) or set(names) != set(lookup):
        raise ValueError("Partition overlap or missing images")
    group_roles = defaultdict(set)
    roles = {n: role for role in ROLES for n in partitions[role]}
    for n, role in roles.items():
        group_roles[audit[n]["group_id"]].add(role)
    if any(len(r) != 1 for r in group_roles.values()):
        raise ValueError("Group leakage")
    if any(roles[p["a"]] != roles[p["b"]] for p in near):
        raise ValueError("Near duplicate candidate leakage")
    if Counter(lookup[n]["label"] for n in partitions["reference"]) != {0: 4, 1: 4}:
        raise ValueError("Unbalanced reference set")


def create(dataset, old_splits, seed, allow_normal_image_groups):
    samples = load_samples(dataset)
    groups, audit, near = build_groups(dataset, samples, allow_normal_image_groups)
    partitions = assign(groups, samples, audit, seed)
    validate(partitions, samples, audit, near)
    old = {
        p.name: json.loads(p.read_text()) for p in sorted(old_splits.glob("reference_split*.json"))
    }
    for s in samples:
        name = s["image"]
        audit[name].update(s)
        audit[name]["uuid"] = Path(name).stem
        audit[name]["stratum"] = stratum(s)
        audit[name]["role"] = next(r for r in ROLES if name in partitions[r])
        audit[name]["historical_exposure"] = {
            k: "reference"
            if name in v.get("reference_pool", [])
            else "historical_development"
            if name in v.get("test", [])
            else "unlisted"
            for k, v in old.items()
        }
    counts = {
        role: {
            "total": len(names),
            "labels": dict(Counter(audit[n]["label"] for n in names)),
            "strata": dict(Counter(audit[n]["stratum"] for n in names)),
            "groups": len({audit[n]["group_id"] for n in names}),
        }
        for role, names in partitions.items()
    }
    inputs = [
        dataset / "labels" / n for n in ("labels.txt", "manifest.jsonl", "uuid_mapping.jsonl")
    ]
    inputs += sorted(old_splits.glob("reference_split*.json"))
    split = {
        "schema": "he-auto-harness-split-v1",
        "seed": seed,
        "status": "frozen_internal_exploratory_split",
        "normal_policy": "image_candidate_groups_patient_unknown"
        if allow_normal_image_groups
        else "whole_source",
        "dataset": "data/unified",
        "partitions": partitions,
        "development": sorted(partitions["search"] + partitions["selection"]),
        "counts": counts,
        "samples": audit,
        "input_sha256": {str(p): digest(p) for p in inputs},
        "generator_sha256": digest(Path(__file__)),
        "policy": {
            "dhash_max_distance": 4,
            "mixed_stratum_groups": "search",
            "test_quota": TEST_QUOTA,
            "reference_quota": REFERENCE_QUOTA,
            "selection_quota": SELECTION_QUOTA,
            "normal_source_group_relaxation": "explicit_user_choice_required",
            "no_predictions_read": True,
            "test_lock": "manifest lock only; not a filesystem access barrier",
        },
        "limitations": [
            "Historical data reused; this is not unseen external validation.",
            "Normal patient IDs unknown; candidate-group isolation does not establish patient independence.",
            "HE unknown patients grouped; overlap with documented patients cannot be fully excluded.",
            "Source-local Bio patient keys are not globally verified identities.",
            "dHash candidates conservatively grouped, not confirmed duplicates; missing other near duplicates remains possible.",
            "Normal source membership is not individual HE-specific exclusion evidence.",
            "No new clinical review; source labels and prior inclusion decisions retained.",
        ],
    }
    return split, {"groups": groups, "near_duplicate_candidates": near}


def write_bundle(output, split, grouping):
    if output.exists():
        raise FileExistsError("Refusing to overwrite split directory")
    output.mkdir(parents=True)
    split_path = output / "split_v1.json"
    split_path.write_text(json_text(split))
    (output / "group_audit.json").write_text(json_text(grouping))
    stream = io.StringIO()
    fields = [
        "uuid",
        "image",
        "role",
        "label",
        "stratum",
        "group_id",
        "sha256",
        "width",
        "height",
        "flags",
        "historical_exposure",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in split["samples"].values():
        writer.writerow(
            {
                k: json.dumps(row[k], ensure_ascii=False)
                if isinstance(row[k], (dict, list))
                else row[k]
                for k in fields
            }
        )
    (output / "sample_audit.csv").write_text(stream.getvalue())
    # Per-role lists expose no labels, but are not permissions-enforced sandboxes.
    for role, names in split["partitions"].items():
        (output / f"{role}_images.txt").write_text("".join(n + "\n" for n in names))
    lock = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "checksums.json").write_text(json_text(lock))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/unified"))
    parser.add_argument("--old-splits", type=Path, default=Path("configs/reference-split"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-normal-image-groups", action="store_true")
    args = parser.parse_args()
    split, grouping = create(
        args.dataset, args.old_splits, args.seed, args.allow_normal_image_groups
    )
    write_bundle(args.output, split, grouping)
    print(json_text(split["counts"]))


if __name__ == "__main__":
    main()
