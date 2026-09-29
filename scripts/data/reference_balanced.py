"""重选八张参考：阳性来源2+2、阴性原类别2+2，保持患者与重复组隔离。

choose_strata 按预定来源／原类别配额选图；create_balanced_split 生成新划分和交集。
main 拒绝覆盖旧配置，便于重放改变参考来源前后的实验。
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from prepare_reference_split import groups_for  # noqa: E402

from he_agent import dump, load_samples  # noqa: E402
from project_paths import load_paths  # noqa: E402
from reference_split import select_references, validate_split  # noqa: E402

# 明确、可复现的来源/类别配额；不使用模型预测或单图loss选择参考。
STRATA = (
    ("HE_DCNN_binary", 1, "original_label", 0, 1),
    ("HE_DCNN_binary", 1, "original_label", 1, 1),
    ("bioengineering_US_binary", 1, "label_ce5", "CE3", 1),
    ("bioengineering_US_binary", 1, "label_ce5", "CE5", 1),
    ("HE_DCNN_binary", 0, "original_label", 3, 2),
    ("HE_DCNN_binary", 0, "original_label", 4, 2),
)


def choose_strata(samples, ids, notes, provenance, old_pool, seed):
    """完整且已记录的单图组才可入池；同层优先复用旧图减少不必要更换。"""
    group_sizes = Counter(ids[row["image"]] for row in samples)
    candidates = sorted(samples, key=lambda row: row["image"])
    random.Random(seed).shuffle(candidates)
    # Python 排序稳定：优先保留旧池，层内仍保留前面固定种子打乱的次序。
    candidates.sort(key=lambda row: row["image"] not in old_pool)
    selected = []
    for source, label, field, value, count in STRATA:
        available = []
        for row in candidates:
            image = row["image"]
            records = provenance.get(image, [])
            if (
                row["source_dataset"] == source
                and row["label"] == label
                and group_sizes[ids[image]] == 1
                and not notes[image]["notes"]
                and records
                and all(record.get(field) == value for record in records)
            ):
                available.append(image)
        if len(available) < count:
            raise ValueError(f"Not enough independent references for {source}/{field}={value}")
        selected.append(available[:count])
    return selected


def create_balanced_split(dataset, base_path, seed=42):
    """生成显式新配置与旧/新测试交集，不修改当前默认划分。"""
    samples = load_samples(dataset)
    lookup = {row["image"]: row for row in samples}
    base = json.loads(base_path.read_text())
    validate_split(base, samples)
    if len(base["reference_pool"]) != 8:
        raise ValueError("Expected eight-reference parent split")
    ids, notes, near = groups_for(dataset, samples)
    if any(ids[image] != base["samples"][image]["group_id"] for image in lookup):
        raise ValueError("Grouping changed; audit the parent split before migration")
    provenance = defaultdict(list)
    mapping_path = dataset / "labels/uuid_mapping.jsonl"
    for line in mapping_path.read_text().splitlines():
        row = json.loads(line)
        provenance[row["image"]].append(row["source_record"])
    groups = choose_strata(samples, ids, notes, provenance, set(base["reference_pool"]), seed)
    he0, he1, bio3, bio5, neg3, neg4 = groups
    order = [he0[0], neg3[0], bio3[0], neg4[0], bio5[0], neg3[1], he1[0], neg4[1]]
    if len(set(order)) != 8:
        raise ValueError("Strata overlap")
    pool = set(order)
    result = copy.deepcopy(base)
    result.update(
        version=4,
        strategy="source-class-balanced-v1",
        seed=seed,
        pool_per_class=4,
        reference_pool=order,
        reference_order=order,
        references=order[:4],
        test=[row["image"] for row in samples if row["image"] not in pool],
        parent_split_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        near_duplicate_candidate_pairs=near,
    )
    result["selection_audit"] = {
        "strata": [dict(source=s, label=y, field=f, value=v, count=n) for s, y, f, v, n in STRATA],
        "policy": "Documented singleton groups only; fixed-seed selection within strata; prefer parent pool; no predictions read",
        "removed": sorted(set(base["reference_pool"]) - pool),
        "added": sorted(pool - set(base["reference_pool"])),
        "provenance_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "clinical_review": "not_clinically_validated",
        "source_balance_scope": "positive sources 2+2, HE negative original classes 2+2; overall sources 6 HE and 2 Bio",
    }
    result["limitations"] = [
        "Only two sources can be sampled without relaxing the current group-isolation policy.",
        "Web normal images form a 100-image source-level group without patient IDs; held entirely in test.",
        "Web positive images form a three-image source-level group; held entirely in test, not treated as independent cases.",
        "Bioengineering has no negative images; 4+4 overall source balance would confound source and label again.",
        "Unknown HE patients remain grouped in test; overlap with documented patients cannot be fully excluded.",
        "DHash and recorded patient keys are conservative heuristics, not a guarantee of independent patients.",
        "HE original classes and Bio CE stages use different schemes; this pool does not cover every morphology.",
        "References partly came from an already-explored test set; use frozen common cohort and later independent data.",
    ]
    result["counts"] = {
        key: dict(
            total=len(result[key]),
            labels=dict(Counter(lookup[x]["label"] for x in result[key])),
            sources=dict(Counter(lookup[x]["source_dataset"] for x in result[key])),
        )
        for key in ("reference_pool", "test")
    }
    validate_split(result, samples)
    for count in (4, 6, 8):
        selected = select_references(result, samples, count)
        if Counter(row["label"] for row in selected) != {0: count // 2, 1: count // 2}:
            raise ValueError("Unbalanced reference prefix")
    # 参考更换会改变测试成员；提前冻结完整交集，防止病例移出造成虚假提升。
    common = [x for x in base["test"] if x in set(result["test"])]
    cohort = dict(
        schema="reference-comparison-cohort-v1",
        policy="Complete old/new test intersection frozen before new inference",
        parent_split_sha256=result["parent_split_sha256"],
        images=common,
        samples={x: dict(label=lookup[x]["label"], sha256=lookup[x]["sha256"]) for x in common},
        counts=dict(total=len(common), labels=dict(Counter(lookup[x]["label"] for x in common))),
    )
    return result, cohort


def main():
    """读取来源配额参数，生成两份新 JSON：参考划分和旧新测试交集，拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--base-split", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    args = parser.parse_args()
    paths = load_paths()
    args.dataset = args.dataset or paths["dataset"]
    args.base_split = args.base_split or paths["base_split"]
    if (
        args.output.resolve() == args.cohort_output.resolve()
        or args.output.exists()
        or args.cohort_output.exists()
    ):
        parser.error(
            "Use two distinct, new output paths; existing configurations are not overwritten"
        )
    result, cohort = create_balanced_split(args.dataset, args.base_split, args.seed)
    for path, value in ((args.output, result), (args.cohort_output, cohort)):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, value)
    print(json.dumps(dict(split=result["counts"], common=cohort["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
