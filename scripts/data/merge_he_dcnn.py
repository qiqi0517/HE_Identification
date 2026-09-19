"""合并 HE_DCNN 原始图片与逐图标签，保留来源并按文件字节去重。"""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main():
    """核对原始标签证据，合并相同字节的图片并写入新的整理目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    root = source / "data"
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} and p.is_file()
    )
    if not files:
        raise ValueError("No source images found")
    digest = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    grouped = defaultdict(list)
    evidence = defaultdict(list)
    for p, h in digest.items():
        grouped[h].append(p)
    prefixes = {"train": "train_pic", "val": "val/pic", "test": "test/pic"}
    for split, prefix in prefixes.items():
        manifest = root / "meta" / f"{split}.txt"
        for line_num, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            name, raw = line.rsplit(maxsplit=1)
            label = int(raw)
            if label not in range(5):
                raise ValueError(f"Invalid label: {manifest}:{line_num}")
            p = root / prefix / name
            if p not in digest:
                raise ValueError(f"Missing image: {p}")
            evidence[digest[p]].append(
                {
                    "type": "manifest",
                    "file": str(manifest.relative_to(source)),
                    "line": line_num,
                    "image": str(p.relative_to(source)),
                    "split": split,
                    "original_label": label,
                }
            )
    for p in files:
        parts = p.relative_to(root).parts
        if len(parts) == 3 and parts[0] == "train" and parts[1] in {"0", "1", "2", "3", "4"}:
            evidence[digest[p]].append(
                {
                    "type": "class_directory",
                    "image": str(p.relative_to(source)),
                    "split": "train",
                    "original_label": int(parts[1]),
                }
            )
    conflicts = {h: ev for h, ev in evidence.items() if len({e["original_label"] for e in ev}) > 1}
    if conflicts:
        raise ValueError(f"Conflicting original labels: {json.dumps(conflicts)}")
    records = []
    for h, paths in sorted(grouped.items()):
        ev = evidence[h]
        raw = ev[0]["original_label"] if ev else None
        # 原类别 0/1/2 属于包虫病；3/4 为非包虫病；缺少证据仍保留 None。
        label = int(raw in (0, 1, 2)) if raw is not None else None
        patient_groups = sorted(
            {
                str(p.parent.relative_to(source))
                for p in paths
                if "patience" in p.relative_to(root).parts
            }
        )
        records.append(
            {
                "image": f"images/{h}{paths[0].suffix.lower()}",
                "sha256": h,
                "label": label,
                "original_label": raw,
                "status": "verified_label" if ev else "unlabeled",
                "source_paths": [str(p.relative_to(source)) for p in paths],
                "original_splits": sorted(
                    {
                        p.relative_to(root).parts[0]
                        if p.relative_to(root).parts[0] != "train_pic"
                        else "train"
                        for p in paths
                    }
                ),
                "patient_groups": patient_groups,
                "label_evidence": ev,
            }
        )
    # Record patient context for review only, never propagate it to image labels.
    group_labels = defaultdict(set)
    for row in records:
        if row["label"] is not None:
            for group in row["patient_groups"]:
                group_labels[group].add(row["label"])
    for row in records:
        if row["label"] is None:
            row["patient_context_labels_not_image_labels"] = {
                g: sorted(group_labels[g]) for g in row["patient_groups"]
            }
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for row in records:
        shutil.copyfile(source / row["source_paths"][0], output / row["image"])
    labeled = [r for r in records if r["label"] is not None]
    unlabeled = [r for r in records if r["label"] is None]
    (output / "labels.txt").write_text("".join(f"{r['image']} {r['label']}\n" for r in labeled))
    (output / "unlabeled.txt").write_text("".join(r["image"] + "\n" for r in unlabeled))
    (output / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    summary = {
        "source": str(source),
        "source_image_files": len(files),
        "unique_images": len(records),
        "duplicate_copies_removed": len(files) - len(records),
        "labeled_images": len(labeled),
        "unlabeled_images": len(unlabeled),
        "binary_counts": dict(Counter(str(r["label"]) for r in labeled)),
        "original_label_counts": dict(Counter(str(r["original_label"]) for r in labeled)),
        "label_mapping": {"0": "non_echinococcosis", "1": "echinococcosis"},
        "original_to_binary": {"0": 1, "1": 1, "2": 1, "3": 0, "4": 0},
        "label_conflicts": 0,
        "cross_split_exact_duplicates": [
            r["image"] for r in records if len(r["original_splits"]) > 1
        ],
        "deduplication": "SHA-256 of original file bytes; not perceptual deduplication",
        "scope": "All image files under repository data/; out/ CAM artifacts excluded",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        f"""# HE_DCNN 合并数据\n\n0 = 非包虫病；1 = 包虫病。原始类别 0/1/2 → 1，3/4 → 0。\n\n- 原始图片文件：{len(files)}；按文件 SHA-256 去重后：{len(records)}。\n- 有明确标签：{len(labeled)}；无逐图标签：{len(unlabeled)}。\n- images/：所有去重图片，字节保持不变，以 SHA-256 命名。\n- labels.txt：有明确标签的图片相对路径与 0/1 标签，空格分隔。\n- unlabeled.txt：无明确逐图标签图片，不可直接用于监督训练或评分。\n- provenance.jsonl：逐图原路径、原类别、原划分、患者目录及标签证据。\n- summary.json：统计和类别映射。\n\n标签仅来自原始 meta 清单、train/类别目录，或与上述图片字节完全一致的副本。同患者额外切面的标签没有自动继承；JSONL 内的患者上下文仅供审核。训练图片未通过文件名猜测患者编号。\n\n合并目录不是新的训练/测试划分。后续应依据患者信息分组，避免同患者跨集合；原划分已保留。仅进行文件级去重，未做近似图像去重、医学标注审核或图内隐私检查。仓库 out/ 下的 CAM 可视化不属于原始数据，未混入。\n"""
    )
    # Verify every output copy, manifest mapping, class count and source coverage.
    assert sum(len(r["source_paths"]) for r in records) == len(files)
    assert len(labeled) + len(unlabeled) == len(records)
    assert len({r["image"] for r in records}) == len(records)
    for r in records:
        assert hashlib.sha256((output / r["image"]).read_bytes()).hexdigest() == r["sha256"]
    assert len((output / "labels.txt").read_text().splitlines()) == len(labeled)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
