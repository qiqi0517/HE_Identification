"""将已获准的 Normal 来源标签并入主标签表，同时保留其证据局限。

read_labels 校验主表；main 合并已授权的 Normal 批次并更新元数据。
会写入指定数据集，数量断言只适用于这批历史资料，不是通用清洗入口。
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_labels(f):
    """读取图片路径与整数标签；重复路径立即触发校验失败。"""
    d = {}
    for line in f.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, label = line.rsplit(None, 1)
        assert path not in d
        d[path] = int(label)
    return d


def main():
    """按既定 219 张数据清单合并 Normal 标签，并更新来源记录与统计。"""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    a = p.parse_args()
    root = a.dataset
    ld = root / "labels"
    labels = read_labels(ld / "labels.txt")
    normal = read_labels(ld / "labels_normal_source.txt")
    # 数量是这批已审核数据的复现检查，不是任意新数据集的通用假设。
    assert len(normal) == 100 and set(normal.values()) == {0}

    for path, label in normal.items():
        assert path not in labels or labels[path] == label
        labels[path] = label

    assert len(labels) == 219 and Counter(labels.values()) == {0: 114, 1: 105}
    assert all((root / path).is_file() for path in labels)
    records = [json.loads(line) for line in (ld / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 219 and {r["image"] for r in records} == set(labels)

    for r in records:
        if r["image"] in normal:
            r.update(
                label=0,
                status="labeled_from_normal_source",
                source_normal_label=0,
                strict_he_label=None,
                label_basis="Source Normal category accepted as binary negative by user; individual HE-specific exclusion not documented",
            )
        assert r["label"] == labels[r["image"]]

    summary = json.loads((root / "summary.json").read_text())
    summary.update(
        binary_labels={"0": 114, "1": 105},
        normal_source_only=0,
        normal_source_included_as_negative=100,
    )
    classes = json.loads((ld / "class_mapping.json").read_text())
    classes["normal_source_label_0"] = (
        "Included in labels.txt as binary 0 by user decision; source Normal category, not individually documented HE exclusion"
    )
    readme = (
        (root / "README.md")
        .read_text()
        .replace(
            "119 张已有明确二分类标签的图片，0=非包虫病，1=包虫病（0 共 14，1 共 105）",
            "219 张二分类标签，0=非包虫病，1=包虫病（0 共 114，1 共 105；其中 100 个 0 来自 Normal 类）",
        )
        .replace(
            "100 张来源 Normal 的标签 0；其包虫病专项排除依据未提供，因此不自动混入主标签表。",
            "100 张来源 Normal 的标签 0，已按用户要求全部并入 labels.txt；本文件仅保留为来源子集，读取主标签后不要再次追加，避免重复计数。包虫病专项排除依据仍未提供。",
        )
    )
    line = "python scripts/data/merge_normal_labels.py --dataset /新的输出目录"

    if line not in readme:
        readme = readme.replace(
            "--unlabeled-output /独立未标注目录\n```",
            "--unlabeled-output /独立未标注目录\n" + line + "\n```",
        )

    updates = {
        ld / "labels.txt": "".join(f"{path} {label}\n" for path, label in sorted(labels.items())),
        ld / "manifest.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        root / "summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        ld / "class_mapping.json": json.dumps(classes, ensure_ascii=False, indent=2) + "\n",
        root / "README.md": readme,
    }

    # 每个文件先写临时副本再替换；这不是多个文件整体的事务。
    for path, text in updates.items():
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text)
        temp.replace(path)

    assert read_labels(ld / "labels.txt") == labels
    print(
        json.dumps(
            {
                "total": 219,
                "positive": 105,
                "negative": 114,
                "normal_source_subset_retained": 100,
                "unlabeled_included": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
