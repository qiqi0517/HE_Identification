"""准备基础分型 Agent 的参考/开发/测试清单；不读取预测、不调用模型。"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from reference_split import validate_typing_split
from typing_agent import load_samples
from typing_protocol import CLASSES


def prepare(dataset, output, seed=42, overwrite=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError("Split output exists; pass --overwrite to replace")
    samples = load_samples(dataset)
    lookup = {s["image"]: s for s in samples}
    metadata = {
        r["image"]: r
        for r in map(json.loads, (dataset / "labels/manifest.jsonl").read_text().splitlines())
    }
    provenance = defaultdict(list)
    for row in map(
        json.loads, (dataset / "labels/legacy_snapshot/uuid_mapping.jsonl").read_text().splitlines()
    ):
        provenance[row["image"]].append(row)
    parents = {name: name for name in lookup}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    seen, hashes, audit = {}, {}, {}
    for sample in samples:
        name = sample["image"]
        row = metadata[name]
        keys = ["bytes:" + row["sha256"], "pixels:" + row["pixel_sha256"]]
        flags = []
        for old in provenance[name]:
            record, source = old["source_record"], old["source_dataset"]
            if source == "HE_DCNN_binary":
                patients = record.get("patient_groups", [])
                keys += ["he:" + Path(k).name for k in patients] or ["he_unknown_patients"]
                if not patients:
                    flags.append("unknown_he_patients_grouped")
            elif source == "bioengineering_US_binary":
                if not record.get("patient_key"):
                    raise ValueError("Missing Bio patient key")
                keys.append("bio:" + record["patient_key"])
                flags.append("patient_id_only_source_local")
            elif record.get("source_id") == "zenodo_7272660":
                if record.get("patient_id"):
                    keys.append("normal:" + str(record["patient_id"]))
                else:
                    flags.append("normal_patient_unknown_image_groups_only")
            else:
                if not record.get("split_group"):
                    raise ValueError("Missing legacy grouping evidence")
                keys.append("web:" + record["split_group"])
        if "cooperating_hospital" in row["sources"]:
            flags.append("hospital_patient_unknown_image_groups_only")
        with Image.open(dataset / name) as im:
            rgb = im.convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if pixel_hash != row["pixel_sha256"]:
                raise ValueError("Pixel hash mismatch")
            gray = im.convert("L").resize((9, 8), Image.Resampling.BICUBIC)
            values = list(gray.tobytes())
            hashes[name] = sum(
                int(values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8)
                for x in range(8)
            )
        for key in keys:
            if key in seen:
                union(name, seen[key])
            seen[key] = name
        audit[name] = dict(group_keys=keys, flags=flags)
    near = []
    names = sorted(lookup)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            distance = (hashes[a] ^ hashes[b]).bit_count()
            if distance <= 4:
                union(a, b)
                near.append(dict(a=a, b=b, distance=distance))
    grouped = defaultdict(list)
    for name in names:
        grouped[find(name)].append(name)
    groups = {
        hashlib.sha256("\n".join(ns).encode()).hexdigest()[:16]: ns for ns in grouped.values()
    }
    randomizer = random.Random(seed)
    gids = sorted(groups)
    randomizer.shuffle(gids)
    references = []
    used = set()
    for label in CLASSES:
        candidates = [
            g for g in gids if len(groups[g]) == 1 and lookup[groups[g][0]]["label"] == label
        ]
        if not candidates:
            raise ValueError(
                f"No isolated singleton reference for {label}; manual group review needed"
            )
        gid = candidates[0]
        used.add(gid)
        references += groups[gid]
    roles = ("search", "selection", "test")
    remaining = len(samples) - len(references)
    targets = dict(zip(roles, (round(remaining * 0.5), round(remaining * 0.2), 0)))
    targets["test"] = remaining - targets["search"] - targets["selection"]
    pool = [g for g in gids if g not in used]
    totals = Counter(
        (lookup[n]["label"], lookup[n]["source_dataset"]) for g in pool for n in groups[g]
    )
    # Seeded group ordering; keep best allocation among local deterministic attempts.
    # No image or group is split to force nominal role counts.
    best = None
    for attempt in range(400):
        order = pool[:]
        randomizer.shuffle(order)
        order.sort(key=lambda g: -len(groups[g]))
        assigned = {r: [] for r in roles}
        counts = {r: Counter() for r in roles}
        sizes = Counter()

        def cost():
            value = 0.0
            for r in roles:
                fraction = targets[r] / remaining
                value += 8 * (sizes[r] - targets[r]) ** 2 / max(targets[r], 1)
                for key, count in totals.items():
                    expected = count * fraction
                    value += (counts[r][key] - expected) ** 2 / max(expected, 1)
            return value

        for gid in order:
            cells = Counter((lookup[n]["label"], lookup[n]["source_dataset"]) for n in groups[gid])
            scores = []
            for r in roles:
                sizes[r] += len(groups[gid])
                counts[r].update(cells)
                scores.append((cost(), r))
                sizes[r] -= len(groups[gid])
                counts[r].subtract(cells)
            role = min(scores)[1]
            assigned[role].append(gid)
            sizes[role] += len(groups[gid])
            counts[role].update(cells)
        present = {
            r: {label for (label, source), count in counts[r].items() if count > 0} for r in roles
        }
        if any(labels != set(CLASSES) for labels in present.values()):
            continue
        candidate = (cost(), assigned)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Unable to cover seven labels in all roles without splitting groups")
    partitions = {"reference": references}
    for r in roles:
        partitions[r] = sorted(n for g in best[1][r] for n in groups[g])
    role_of = {n: r for r, ns in partitions.items() for n in ns}
    gid_of = {n: g for g, ns in groups.items() for n in ns}
    split = {
        "schema": "he-typing-split-v1",
        "seed": seed,
        "dataset": str(dataset),
        "dataset_labels_sha256": hashlib.sha256(
            (dataset / "labels/labels_typing.txt").read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partitions": partitions,
        "samples": {
            n: dict(
                label=s["label"],
                sha256=s["sha256"],
                group_id=gid_of[n],
                role=role_of[n],
                source_dataset=s["source_dataset"],
                **audit[n],
            )
            for n, s in lookup.items()
        },
        "target_counts": {"reference": 7, **targets},
        "counts": {
            r: {
                "total": len(ns),
                "labels": dict(Counter(lookup[n]["label"] for n in ns)),
                "sources": dict(Counter(lookup[n]["source_dataset"] for n in ns)),
                "groups": len({gid_of[n] for n in ns}),
            }
            for r, ns in partitions.items()
        },
        "near_duplicate_candidates": near,
        "limitations": [
            "Internal exploratory split; historical samples already used in development",
            "Hospital/Normal patient IDs unknown; image groups do not establish patient independence",
            "dHash <= 4 candidates kept together, not clinically confirmed duplicates",
            "No image text/identity redaction or clinical relabeling performed",
            "Group integrity takes precedence over exact target counts",
        ],
    }
    validate_typing_split(split, samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", type=Path, default=root / "data/unified_typing_v2")
    parser.add_argument(
        "--output", type=Path, default=root / "configs/auto-harness-typing/split_v1.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.dataset, args.output, args.seed, args.overwrite)["counts"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
