"""推理时校验固定划分并选取参考示例；不生成或改写划分文件。

validate_split 返回冻结测试图与默认示例；select_references 扩展示例前缀。
生成与重选划分在 scripts/data/，此文件不能按模型成绩修改参考池。
"""

import random
from collections import Counter


def validate_split(split, samples):
    """检查覆盖范围、标签与哈希、示例数量和分组隔离，返回测试图及参考图。"""
    lookup = {s["image"]: s for s in samples}
    pool = split["reference_pool"]
    test = split["test"]
    refs = split["references"]
    if (
        len(set(pool)) != len(pool)
        or len(set(test)) != len(test)
        or set(pool) & set(test)
        or set(pool) | set(test) != set(lookup)
    ):
        raise ValueError("Split overlap, duplicate, or missing sample")
    if len(refs) != 4 or len(set(refs)) != 4 or not set(refs) <= set(pool):
        raise ValueError("Need four distinct pool references")
    if Counter(lookup[x]["label"] for x in refs) != {0: 2, 1: 2}:
        raise ValueError("References must be 2 negative + 2 positive")
    # v1 是历史不平衡池；新版本才要求池内两类配额相等，仍允许旧实验重放。
    if split.get("version", 1) >= 2:
        quota = split.get("pool_per_class")
        if type(quota) is not int or quota < 2:
            raise ValueError("Invalid balanced pool quota")
        if Counter(lookup[x]["label"] for x in pool) != {0: quota, 1: quota}:
            raise ValueError("Reference pool does not match balanced class quota")
    order = split.get("reference_order")
    if order is not None:
        if len(order) != len(pool) or len(set(order)) != len(order) or set(order) != set(pool):
            raise ValueError("Reference order must cover the pool exactly once")
        if order[:4] != refs:
            raise ValueError("Reference order must preserve the default four")
        # 每个偶数前缀都保持类别均衡，4/6/8 图实验只增示例而不换原有顺序。
        for count in range(4, len(order) + 1, 2):
            if Counter(lookup[x]["label"] for x in order[:count]) != {0: count // 2, 1: count // 2}:
                raise ValueError("Reference order must have balanced even prefixes")
    for image, s in lookup.items():
        saved = split["samples"][image]
        if saved["sha256"] != s["sha256"] or saved["label"] != s["label"]:
            raise ValueError("Stale split: dataset changed")
    # 同一患者/来源/重复组不能一部分作参考、一部分被评价。
    if {split["samples"][x]["group_id"] for x in pool} & {
        split["samples"][x]["group_id"] for x in test
    }:
        raise ValueError("Group leakage")
    if not test:
        raise ValueError("Empty test set")
    return [lookup[x] for x in test], [lookup[x] for x in refs]


def select_references(split, samples, count=None):
    """默认复用四张示例；扩展时按固定种子各加同量两类候选。

    示例仅从已冻结参考池选取。增加数量不会改变测试集或前四张顺序。
    """
    _, default = validate_split(split, samples)
    if count is None:
        count = 4
    if type(count) is not int or count < 4 or count % 2:
        raise ValueError("Reference count must be an even integer >= 4")
    lookup = {s["image"]: s for s in samples}
    if "reference_order" in split:
        if count > len(split["reference_order"]):
            raise ValueError("Insufficient balanced pool candidates for requested reference count")
        return [lookup[x] for x in split["reference_order"][:count]]
    extra = {0: [], 1: []}
    for image in sorted(set(split["reference_pool"]) - set(split["references"])):
        extra[lookup[image]["label"]].append(lookup[image])
    needed = (count - 4) // 2
    if any(len(extra[y]) < needed for y in (0, 1)):
        raise ValueError("Insufficient balanced pool candidates for requested reference count")
    rng = random.Random(split.get("seed", 42))
    for y in (0, 1):
        rng.shuffle(extra[y])
    return default + [extra[y][i] for i in range(needed) for y in (0, 1)]


def validate_typing_split(split, samples, role="search", reference_count=None):
    """读取新分型划分；不转换二分类标签，也不静默剔除过期清单图片。"""
    if split.get("schema") != "he-typing-split-v1":
        raise ValueError("Use a he-typing-split-v1 manifest; binary splits are incompatible")
    roles = ("reference", "search", "selection", "test")
    if role not in roles[1:] or set(split["partitions"]) != set(roles):
        raise ValueError("Invalid typing partition")
    lookup = {s["image"]: s for s in samples}
    saved = split["samples"]
    seen, group_roles = set(), {}
    for name in roles:
        names = split["partitions"][name]
        if not isinstance(names, list) or not names:
            raise ValueError("Empty or invalid partition")
        for image in names:
            if image in seen or image not in lookup or image not in saved:
                raise ValueError("Split overlap or unknown/excluded image")
            seen.add(image)
            before, now = saved[image], lookup[image]
            if before.get("label") != now["label"] or before.get("sha256") != now["sha256"]:
                raise ValueError("Stale typing split: label/hash differs")
            gid = before.get("group_id")
            if not isinstance(gid, str) or not gid or group_roles.get(gid, name) != name:
                raise ValueError("Typing group leakage")
            group_roles[gid] = name
    if seen != set(lookup) or set(saved) != seen:
        raise ValueError("Typing split does not cover the current dataset")
    refs = split["partitions"]["reference"]
    if reference_count is not None:
        if type(reference_count) is not int or not 1 <= reference_count <= len(refs):
            raise ValueError("Reference count must be within the frozen reference pool")
        refs = refs[:reference_count]
    return [lookup[n] for n in split["partitions"][role]], [lookup[n] for n in refs]
