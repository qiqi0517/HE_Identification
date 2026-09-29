"""离线按患者、来源和重复图像分组，生成固定参考池与测试集。

groups_for 构建不可拆分组；balanced_groups 按完整组凑齐配额；
create_split 生成元数据；main 只写新的划分文件，不处理模型预测。
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# 支持从任意工作目录直接执行脚本，定位项目中的共用数据校验逻辑。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from he_agent import dump, load_samples  # noqa: E402
from project_paths import load_paths  # noqa: E402
from reference_split import validate_split  # noqa: E402


def groups_for(dataset, samples):
    """根据患者/来源关联、相同像素及近重复候选构建不可拆分的图片组。"""
    mapping = defaultdict(list)
    for line in (dataset / "labels/uuid_mapping.jsonl").read_text().splitlines():
        r = json.loads(line)
        mapping[r["image"]].append(r["source_record"])
    parent = {s["image"]: s["image"] for s in samples}

    def find(x):
        """查找并查集根节点；路径压缩减少后续查找次数。"""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        """合并关联图片，保证它们不会跨参考池和测试集。"""
        parent[find(x)] = find(y)

    seen = {}
    descriptors = {}
    hashes = {}
    for s in samples:
        image = s["image"]
        tokens = []
        reasons = []
        for r in mapping[image]:
            source = s["source_dataset"]
            if source == "HE_DCNN_binary":
                patients = r.get("patient_groups", [])
                tokens.extend(source + ":patient:" + Path(p).name for p in patients)
                if not patients:
                    tokens.append(source + ":unknown_patients")
                    reasons.append("unknown patient: grouped within source")
            elif source == "bioengineering_US_binary":
                if not r.get("patient_key"):
                    raise ValueError("Missing Bioengineering patient key")
                tokens.append(source + ":" + r["patient_key"])
            else:
                tokens.append(source + ":" + r["source_id"])
                reasons.append("conservative source-level group")
        if not tokens:
            raise ValueError("Missing provenance")
        with Image.open(dataset / image) as im:
            rgb = im.convert("RGB")
            tokens.append(
                "pixels:" + hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            )
            # dHash 缩略图仅用于分组比较，不改变保存或上传的原图。
            gray = im.convert("L").resize((9, 8))
            v = list(gray.getdata())
            bits = [v[y * 9 + x] > v[y * 9 + x + 1] for y in range(8) for x in range(8)]
            hashes[image] = sum(int(b) << i for i, b in enumerate(bits))
        for t in tokens:
            if t in seen:
                union(image, seen[t])
            else:
                seen[t] = image
        descriptors[image] = {"provenance_group_keys": tokens[:-1], "notes": reasons}
    near = []
    names = sorted(parent)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            if (hashes[x] ^ hashes[y]).bit_count() <= 4:
                union(x, y)
                near.append([x, y])
    grouped = defaultdict(list)
    for x in names:
        grouped[find(x)].append(x)
    ids = {
        x: hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()[:16]
        for xs in grouped.values()
        for x in xs
    }
    return ids, descriptors, near


def balanced_groups(groups, eligible, mandatory, pool_per_class):
    """以候选顺序为优先级，精确凑齐两类配额；绝不拆分关联图片组。

    动态规划保留每种可达计数的首个组合，不依据模型成绩挑选参考图。
    无法同时满足配额和分组隔离时明确失败。
    """
    if type(pool_per_class) is not int or pool_per_class < 2:
        raise ValueError("pool_per_class must be an integer >= 2")
    if not mandatory <= set(eligible):
        raise ValueError("Preserved references include ineligible groups")
    initial = Counter(s["label"] for g in mandatory for s in groups[g])
    if any(initial[y] > pool_per_class for y in (0, 1)):
        raise ValueError("Preserved reference groups exceed the pool quota")
    states = {(initial[0], initial[1]): tuple(sorted(mandatory))}
    target = (pool_per_class, pool_per_class)
    for g in eligible:
        if g in mandatory:
            continue
        counts = Counter(s["label"] for s in groups[g])
        # 遍历上一轮状态的副本，防止同一不可拆组被重复选入（0/1 背包思路）。
        for (n0, n1), selected in list(states.items()):
            key = (n0 + counts[0], n1 + counts[1])
            if max(key) <= pool_per_class and key not in states:
                states[key] = (*selected, g)
        if target in states:
            break
    if target not in states:
        raise ValueError("Cannot build a balanced pool without splitting patient/source groups")
    return set(states[target])


def create_split(dataset, seed=42, pool_per_class=4, base_split=None):
    """创建小型平衡参考池；迁移时保留旧划分的四张默认示例及顺序。"""
    samples = load_samples(dataset)
    previous = None
    if base_split is not None:
        previous = json.loads(base_split.read_text())
        validate_split(previous, samples)
    ids, notes, near = groups_for(dataset, samples)
    groups = defaultdict(list)
    for s in samples:
        groups[ids[s["image"]]].append(s)
    rng = random.Random(seed)
    candidates = sorted(groups)
    rng.shuffle(candidates)
    # 未知患者或保守来源组整组留在测试集，不为凑配额而拆开。
    eligible = [g for g in candidates if all(not notes[s["image"]]["notes"] for s in groups[g])]
    mandatory = set()
    if previous is not None:
        mandatory = {ids[x] for x in previous["references"]}
        old_pool = set(previous["reference_pool"])
        # 优先复用旧池的候选；同优先级内仍由固定随机种子决定。
        eligible.sort(key=lambda g: not all(s["image"] in old_pool for s in groups[g]))
    chosen = balanced_groups(groups, eligible, mandatory, pool_per_class)
    pool = [s for s in samples if ids[s["image"]] in chosen]
    test = [s for s in samples if ids[s["image"]] not in chosen]
    # 优先不同来源/阶段，仅使用来源资料，不按模型预测或测试得分挑选。
    provenance = {}
    for line in (dataset / "labels/uuid_mapping.jsonl").read_text().splitlines():
        r = json.loads(line)
        provenance[r["image"]] = r["source_record"]
    refs = []
    for label in (0, 1):
        available = [s for s in pool if s["label"] == label]
        rng.shuffle(available)
        used = set()
        for _ in range(2):
            s = next(
                (
                    s
                    for s in available
                    if (s["source_dataset"], provenance[s["image"]].get("label_ce5")) not in used
                ),
                available[0],
            )
            refs.append(s["image"])
            used.add((s["source_dataset"], provenance[s["image"]].get("label_ce5")))
            available.remove(s)
    rng.shuffle(refs)
    if previous is not None:
        refs = list(previous["references"])
    result = {
        "version": 2,
        "pool_per_class": pool_per_class,
        "seed": seed,
        "policy": "patient/source and exact pixels grouped; dHash distance <=4 grouped; unknown patients held in test",
        "reference_pool": [s["image"] for s in pool],
        "references": refs,
        "test": [s["image"] for s in test],
        "samples": {
            s["image"]: {
                "label": s["label"],
                "sha256": s["sha256"],
                "group_id": ids[s["image"]],
                **notes[s["image"]],
            }
            for s in samples
        },
        "near_duplicate_candidate_pairs": near,
        "limitations": [
            "Unknown HE_DCNN patients can still overlap documented patients; source-local IDs are not globally verified.",
            "dHash is a conservative heuristic, not proof of absent near duplicates.",
            "Normal cohort held entirely in test; references do not include normal-source controls.",
        ],
        "counts": {
            k: {"total": len(xs), "labels": dict(Counter(s["label"] for s in xs))}
            for k, xs in [("reference_pool", pool), ("test", test)]
        },
    }
    if previous is not None:
        result["parent_split_sha256"] = hashlib.sha256(base_split.read_bytes()).hexdigest()
    validate_split(result, samples)
    return result


def main():
    """生成新的固定划分文件；拒绝覆盖已有实验划分。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-per-class", type=int, default=4, help="Pool images per class (>= 2)")
    p.add_argument("--base-split", type=Path, help="Preserve this split's four default references")
    a = p.parse_args()
    paths = load_paths()
    a.dataset = a.dataset or paths["dataset"]
    a.output = a.output or paths["split_output"]
    if a.output.exists():
        p.error("Refusing to overwrite fixed split")
    result = create_split(a.dataset, a.seed, a.pool_per_class, a.base_split)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    dump(a.output, result)
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
