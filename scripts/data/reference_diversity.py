"""建立来源覆盖对照：替换两张重复阳性示例，不改变默认四张或阴性示例。

choose_additions 选择来源补充；create_diverse_split 保留旧四图并建立新划分和交集。
用于历史来源对照，不是运行 Agent 时的动态检索器。
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 推理模块保持在 src；本文件及分组生成器只在准备实验配置时运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from prepare_reference_split import groups_for  # noqa: E402

from he_agent import dump, load_samples  # noqa: E402
from project_paths import load_paths  # noqa: E402
from reference_split import select_references, validate_split  # noqa: E402


def choose_additions(samples, ids, notes, provenance, old_pool, seed):
    """只根据来源、原始标签、完整独立组和种子选图，不读取预测分数。

    本对照只更换两张图，故候选限定为已知关联规则下的单图组；不拆分大组。
    """
    grouped = defaultdict(list)
    for row in samples:
        grouped[ids[row["image"]]].append(row)
    candidates = sorted(grouped)
    random.Random(seed).shuffle(candidates)
    selected = []
    for original_label in (0, 1):
        for group in candidates:
            rows = grouped[group]
            if len(rows) != 1:
                continue
            row = rows[0]
            name = row["image"]
            if (
                row["source_dataset"] != "HE_DCNN_binary"
                or row["label"] != 1
                or name in old_pool
                or notes[name]["notes"]
            ):
                continue
            records = provenance.get(name, [])
            if not records or any(r.get("original_label") != original_label for r in records):
                continue
            selected.append(name)
            break
        else:
            raise ValueError(
                f"No independent HE_DCNN candidate for original label {original_label}"
            )
    return selected


def create_diverse_split(dataset, base_path, seed=42):
    """从八张 4+4 旧池生成受控变体，以及冻结的新旧共同测试清单。"""
    samples = load_samples(dataset)
    lookup = {r["image"]: r for r in samples}
    base = json.loads(base_path.read_text())
    validate_split(base, samples)
    if len(base["reference_pool"]) != 8:
        raise ValueError("This controlled comparison requires an eight-image base pool")
    old_order = [r["image"] for r in select_references(base, samples, 8)]
    old_negatives = [x for x in old_order if lookup[x]["label"] == 0]
    if len(old_negatives) != 4:
        raise ValueError("Base pool must contain four negatives")
    fixed = base["references"]
    replacements = [x for x in old_order if lookup[x]["label"] == 1 and x not in fixed]
    if len(replacements) != 2:
        raise ValueError("Need two replaceable positive references outside the default four")
    provenance = defaultdict(list)
    mapping_path = dataset / "labels/uuid_mapping.jsonl"
    for line in mapping_path.read_text().splitlines():
        row = json.loads(line)
        provenance[row["image"]].append(row["source_record"])
    # 限定基线内容，防止把任意旧池误称为此次“去除重复 CE3/CE5”的对照。
    for collection in (replacements, [x for x in fixed if lookup[x]["label"] == 1]):
        stages = []
        for name in collection:
            if lookup[name]["source_dataset"] != "bioengineering_US_binary":
                raise ValueError("Expected Bioengineering positives in this base design")
            values = {r.get("label_ce5") for r in provenance[name]}
            if len(values) != 1:
                raise ValueError("Missing or conflicting stage provenance")
            stages.extend(values)
        if set(stages) != {"CE3", "CE5"}:
            raise ValueError("Expected one CE3 and one CE5 in each base positive pair")
    ids, notes, near = groups_for(dataset, samples)
    if any(ids[x] != base["samples"][x]["group_id"] for x in lookup):
        raise ValueError("Grouping changed; rebuild and audit the base split first")
    additions = choose_additions(samples, ids, notes, provenance, set(old_order), seed)
    new_pool = (set(old_order) - set(replacements)) | set(additions)
    extra_negatives = [x for x in old_negatives if x not in fixed]
    order = list(fixed) + [x for pair in zip(extra_negatives, additions) for x in pair]
    result = copy.deepcopy(base)
    result.update(
        version=3,
        seed=seed,
        pool_per_class=4,
        strategy="source-diversity-controlled-v1",
        reference_pool=[r["image"] for r in samples if r["image"] in new_pool],
        reference_order=order,
        test=[r["image"] for r in samples if r["image"] not in new_pool],
        parent_split_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        near_duplicate_candidate_pairs=near,
        selection_audit={
            "rules": "Keep default four and all negatives; replace spare CE3/CE5 with HE original classes 0/1; singleton documented groups; fixed seed; no predictions read",
            "removed": replacements,
            "added": additions,
            "provenance_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            "added_original_labels": [0, 1],
            "clinical_review": "not_clinically_validated",
        },
    )
    result["limitations"] = list(base.get("limitations", [])) + [
        "New references previously belonged to the exploratory test set; this is not fresh external validation.",
        "Only two HE original classes are added; this is not full disease/stage coverage.",
        "Normal-source images remain outside the reference pool; source bias is reduced in design, not proven eliminated.",
    ]
    result["counts"] = {
        key: {
            "total": len(result[key]),
            "labels": dict(Counter(lookup[x]["label"] for x in result[key])),
        }
        for key in ("reference_pool", "test")
    }
    validate_split(result, samples)
    common = [x for x in base["test"] if x in set(result["test"])]
    cohort = {
        "schema": "reference-comparison-cohort-v1",
        "policy": "Intersection of complete old/new test sets, frozen before new inference",
        "parent_split_sha256": result["parent_split_sha256"],
        "images": common,
        "samples": {
            x: {"label": lookup[x]["label"], "sha256": lookup[x]["sha256"]} for x in common
        },
        "counts": {
            "total": len(common),
            "labels": dict(Counter(lookup[x]["label"] for x in common)),
        },
    }
    return result, cohort


def main():
    """显式创建新配置；拒绝覆盖已有文件，不修改默认划分和历史运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--base-split", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = load_paths()
    args.dataset = args.dataset or paths["dataset"]
    args.base_split = args.base_split or paths["base_split"]
    if args.output.resolve() == args.cohort_output.resolve():
        parser.error("Split and cohort outputs must be different files")
    if args.output.exists() or args.cohort_output.exists():
        parser.error("Refusing to overwrite an existing split or cohort")
    split, cohort = create_diverse_split(args.dataset, args.base_split, args.seed)
    for path, value in ((args.output, split), (args.cohort_output, cohort)):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, value)
    print(json.dumps({"split": split["counts"], "comparison": cohort["counts"]}))


if __name__ == "__main__":
    main()
