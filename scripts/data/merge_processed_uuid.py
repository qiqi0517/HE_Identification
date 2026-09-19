"""按固定 UUID 合并已整理图片；未标注图片保存到独立目录。"""

import argparse
import hashlib
import json
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def read_labels(f):
    """读取可选标签表；文件不存在时返回空字典。"""
    d = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            path, label = line.rsplit(None, 1)
            assert path not in d
            d[path] = int(label)
    return d


def write_json(f, x):
    """保存可读 JSON 元数据。"""
    f.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(f, x):
    """逐行保存来源记录，保留中文说明。"""
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in x))


def main():
    """合并三个既定来源，保留原图字节并将未标注样本单独存放。"""
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unlabeled-output", type=Path, required=True)
    a = p.parse_args()
    base = a.project / "data/processed"
    out = a.output
    assert not out.exists(), f"Refusing overwrite: {out}"
    assert not a.unlabeled_output.exists(), "Refusing overwrite of unlabeled output"
    sources = [
        base / n
        for n in ["HE_DCNN_binary", "bioengineering_US_binary", "web_liver_ultrasound_20260918"]
    ]
    namespace = uuid.UUID("2a665fd0-9d69-4d8c-8a8a-390209baf28d")
    images = {}
    mapping = []
    excluded = []
    source_counts = {}

    for source in sources:
        labels = read_labels(source / "labels.txt")
        normal = read_labels(source / "labels_normal_source.txt")
        ce5 = read_labels(source / "labels_ce5.txt")
        unlab = (
            set((source / "unlabeled.txt").read_text().splitlines())
            if (source / "unlabeled.txt").exists()
            else set()
        )
        assert not (set(labels) & set(normal) or set(labels) & unlab or set(normal) & unlab)
        prov = [json.loads(line) for line in (source / "provenance.jsonl").read_text().splitlines()]
        indexed = {r.get("image", r.get("path")): r for r in prov}
        selected = set(labels) | set(normal) | unlab
        excluded_paths = set()
        for r in prov:
            old = r.get("image", r.get("path"))
            if old not in selected:
                assert r.get("included") is False or old.startswith("review/"), old
                excluded_paths.add(old)
                excluded.append(
                    {
                        "source_dataset": source.name,
                        "source_image": str((source / old).relative_to(a.project)),
                        "source_record": r,
                    }
                )
        actual = {
            str(f.relative_to(source))
            for f in source.rglob("*")
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        }
        assert actual == selected | excluded_paths, (
            f"Unaccounted image files in {source.name}: {actual ^ (selected | excluded_paths)}"
        )
        source_counts[source.name] = len(selected)
        for old in sorted(selected):
            f = source / old
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            # 固定命名空间 + 文件哈希使同一原图重跑得到相同 UUID。
            uid = str(uuid.uuid5(namespace, sha))
            ext = f.suffix.lower()
            new = f"images/{uid}{ext}"
            source_record = indexed[old]
            expected = source_record.get("sha256", source_record.get("file_sha256"))
            assert expected == sha
            label = labels.get(old)
            proxy = normal.get(old)
            status = (
                "labeled"
                if label is not None
                else "normal_source_only"
                if proxy is not None
                else "unlabeled"
            )
            assert label in (None, 0, 1) and proxy in (None, 0)
            with Image.open(f) as im:
                im.load()
                rgb = im.convert("RGB")
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                w, h = im.size
            m = {
                "uuid": uid,
                "image": new,
                "source_dataset": source.name,
                "old_image": str(f.relative_to(a.project)),
                "sha256": sha,
                "source_record": source_record,
            }
            mapping.append(m)
            if sha in images:
                assert (
                    images[sha]["label"] == label and images[sha]["source_normal_label"] == proxy
                ), "Duplicate label conflict"
                images[sha]["source_paths"].append(m["old_image"])
                continue
            images[sha] = {
                "uuid": uid,
                "image": new,
                "label": label,
                "source_normal_label": proxy,
                "status": status,
                "ce5_label": ce5.get(old),
                "sha256": sha,
                "pixel_sha256": pixels,
                "width": w,
                "height": h,
                "source_paths": [m["old_image"]],
                "copy_from": str(f),
                "source_dataset": source.name,
            }

    assert len(mapping) == 256 and len(excluded) == 8
    # 未标注图片不进入统一监督数据集，也不根据同患者或来源自动补标签。
    unlabeled_records = [r for r in images.values() if r["status"] == "unlabeled"]
    unlabeled_ids = {r["uuid"] for r in unlabeled_records}
    unlabeled_mapping = [r for r in mapping if r["uuid"] in unlabeled_ids]
    mapping = [r for r in mapping if r["uuid"] not in unlabeled_ids]
    images = {k: r for k, r in images.items() if r["uuid"] not in unlabeled_ids}
    separate = a.unlabeled_output
    separate.mkdir(parents=True)
    (separate / "images").mkdir()

    for r in unlabeled_records:
        f = Path(r.pop("copy_from"))
        t = separate / r["image"]
        shutil.copy2(f, t)
        assert hashlib.sha256(t.read_bytes()).hexdigest() == r["sha256"]

    write_jsonl(separate / "manifest.jsonl", unlabeled_records)
    write_jsonl(separate / "uuid_mapping.jsonl", unlabeled_mapping)
    write_json(
        separate / "summary.json",
        {
            "image_files": len(unlabeled_records),
            "included_in_unified_dataset": False,
            "label": None,
        },
    )
    (separate / "README.md").write_text(
        "37 张未标注图片，UUID 命名，保留原始字节。本目录独立于 unified 数据集；不得自动赋予 0 或 1 标签。manifest.jsonl 和 uuid_mapping.jsonl 保存原路径及来源信息。\n"
    )
    out.mkdir(parents=True)
    (out / "images").mkdir()
    (out / "labels").mkdir()

    for r in images.values():
        src = Path(r.pop("copy_from"))
        dst = out / r["image"]
        shutil.copy2(src, dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == r["sha256"]

    records = sorted(images.values(), key=lambda r: r["uuid"])
    labeldir = out / "labels"
    (labeldir / "labels.txt").write_text(
        "".join(f"{r['image']} {r['label']}\n" for r in records if r["label"] is not None)
    )
    (labeldir / "labels_normal_source.txt").write_text(
        "".join(
            f"{r['image']} {r['source_normal_label']}\n"
            for r in records
            if r["status"] == "normal_source_only"
        )
    )
    assert all(r["status"] != "unlabeled" for r in records)
    (labeldir / "labels_ce5.txt").write_text(
        "".join(f"{r['image']} {r['ce5_label']}\n" for r in records if r["ce5_label"] is not None)
    )
    write_jsonl(labeldir / "manifest.jsonl", records)
    write_jsonl(labeldir / "uuid_mapping.jsonl", mapping)
    write_jsonl(labeldir / "excluded_records.jsonl", excluded)
    write_json(
        labeldir / "class_mapping.json",
        {
            "binary": {"0": "non-echinococcosis", "1": "echinococcosis"},
            "normal_source_label_0": "Source Normal category only; HE-specific exclusion not documented",
            "ce5": {"0": "CE1", "1": "CE2", "2": "CE3", "3": "CE4", "4": "CE5"},
            "ce5_scope": "Only original Bioengineering labels_ce5.txt; not inferred for other datasets",
            "unlabeled": None,
        },
    )
    pixels = defaultdict(list)

    for r in records:
        pixels[r["pixel_sha256"]].append(r["image"])

    write_json(labeldir / "pixel_duplicate_groups.json", [v for v in pixels.values() if len(v) > 1])
    summary = {
        "input_images": len(mapping) + len(unlabeled_records),
        "unlabeled_separate_images": len(unlabeled_records),
        "output_images": len(records),
        "exact_byte_duplicates_merged": len(mapping) - len(records),
        "pixel_duplicate_groups": sum(len(v) > 1 for v in pixels.values()),
        "source_images": source_counts,
        "binary_labels": dict(Counter(str(r["label"]) for r in records if r["label"] is not None)),
        "normal_source_only": sum(r["status"] == "normal_source_only" for r in records),
        "unlabeled": sum(r["status"] == "unlabeled" for r in records),
        "ce5_labeled": sum(r["ce5_label"] is not None for r in records),
        "excluded_review_records": len(excluded),
        "uuid_version": 5,
        "uuid_namespace": str(namespace),
        "uuid_input": "image file SHA256",
        "image_bytes_unchanged": True,
    }
    write_json(out / "summary.json", summary)
    (out / "README.md").write_text("""# 统一 UUID 肝脏超声数据

三个已整理来源合并为一个 images/ 和一个 labels/。原有 processed 来源目录保持不变。只合并已入选且有标签的图片；37 张未标注图片移入独立的 unlabeled_uuid 输出；原 review 的 8 条记录继续排除，未把 CT、胸片或标签冲突图加入模型输入。

## 目录和标签

- images/：219 张图像，文件名为 UUID，保留原扩展名和原始文件字节。
- labels/labels.txt：119 张已有明确二分类标签的图片，0=非包虫病，1=包虫病（0 共 14，1 共 105）。每行 `images/UUID.jpg 0或1`。
- labels/labels_normal_source.txt：100 张来源 Normal 的标签 0；其包虫病专项排除依据未提供，因此不自动混入主标签表。
- 未标注的 37 张图片不在本数据集内，单独存于 processed/unlabeled_uuid/images/，该目录有独立溯源清单。
- labels/labels_ce5.txt：保留 Bioengineering 的 80 个分期标签，CE1=0、CE2=1、CE3=2、CE4=3、CE5=4。这不是二分类标签。
- labels/manifest.jsonl：所有图片的 UUID、相对路径、标签状态、尺寸与哈希。
- labels/uuid_mapping.jsonl：旧路径到新 UUID 的映射，嵌入原来源记录，包括原划分、病例关联和诊断依据等可用信息。
- labels/excluded_records.jsonl：8 条既有 review 排除记录，原图仍在原来源目录。

所有标签中的图片路径都相对于本 unified 目录，而不是 labels 目录。读取时使用 `unified目录 / 行内图片路径`。

## 命名和校验

UUIDv5 由固定命名空间与原文件 SHA256 生成，相同文件重跑得到相同 UUID。名称不编码类别或疾病分期。精确字节重复可合并并保留多条来源，出现标签冲突则停止；本次无字节重复。全部图像可解码，复制后的 SHA256 与源文件一致。未缩放、裁剪、重新编码或增强。相同解码像素的组另记录在 labels/pixel_duplicate_groups.json；不能据此排除近重复。

原图中的文字、箭头或拼图不会因 UUID 命名而消失，本次没有重新审查或修改已有来源的图像内容。仍需按原来源说明评估质量和许可。图注和标签元数据用于溯源，不作为模型诊断输入。尚未划分 train/val/test；训练需根据保留的患者/来源关联防止数据泄漏。

## 复现

使用含 Pillow 的 Python：

```sh
python scripts/data/merge_processed_uuid.py --project /Users/qiqi/Research/HE_Identification --output /新的输出目录 --unlabeled-output /独立未标注目录
```

脚本仅读取明确列出的三个来源，拒绝覆盖已有输出，不会把 unified 自身再次合并。
""")
    main = read_labels(labeldir / "labels.txt")
    normal = read_labels(labeldir / "labels_normal_source.txt")
    unl = set()
    assert len(main) == 119 and Counter(main.values()) == {1: 105, 0: 14}
    assert len(normal) == 100 and len(unlabeled_records) == 37 and len(records) == 219
    assert not (set(main) & set(normal) or set(main) & unl or set(normal) & unl)
    assert set(main) | set(normal) | unl == {r["image"] for r in records}
    assert all((out / x).is_file() for x in set(main) | set(normal) | unl)
    assert all(uuid.UUID(r["uuid"]).version == 5 for r in records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
