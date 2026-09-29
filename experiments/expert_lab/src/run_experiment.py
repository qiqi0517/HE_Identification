"""独立实验：只编码固定参考池，生成可供未来 Agent 使用的检索工具结果。"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from experiments.expert_lab.src.ultrasound_encoder import (
    OFFICIAL_SHA256,
    UltrasoundEncoder,
    file_sha256,
    nearest_references,
)


def validated_reference_pool(data_root: Path, split_path: Path) -> tuple[list[str], dict, dict]:
    """拒绝无标签图、重复项、路径越界以及参考池与测试集交叉。"""
    split = json.loads(split_path.read_text())
    names, tests = split["reference_pool"], split["test"]
    if not names or len(names) != len(set(names)) or len(tests) != len(set(tests)):
        raise ValueError("Empty reference pool or duplicate split entries")
    if set(names) & set(tests) or not set(split["references"]) <= set(names):
        raise ValueError("Split overlap or references outside the reference pool")
    labels = {}
    for line in (data_root / "labels/labels.txt").read_text().splitlines():
        name, value = line.rsplit(maxsplit=1)
        if value not in ("0", "1") or name in labels:
            raise ValueError("Invalid/duplicate diagnostic label")
        labels[name] = int(value)
    # 测试真值只存在本地标签表，既不参与模型输入，也不参与近邻选择。
    reference_labels = {name: labels[name] for name in names}
    for name in names:
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise ValueError("Reference image missing or outside data root")
    return names, reference_labels, split


def run(data_root: Path, split_path: Path, checkpoint: Path, output: Path) -> dict:
    """执行真实预训练模型前向；无训练、无测试集评分、无大模型 API 调用。"""
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous experiment records")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    start = time.perf_counter()
    names, labels, split = validated_reference_pool(data_root, split_path)
    before = {name: file_sha256(data_root / name) for name in names}
    encoder = UltrasoundEncoder(checkpoint)
    features = np.stack([encoder.encode(data_root / name) for name in names])
    elapsed = time.perf_counter() - start
    tools = []
    for query in split["references"]:
        neighbors = nearest_references(features[names.index(query)], features, names, query)
        for item in neighbors:
            item["reference_label"] = labels[item["image"]]
        tools.append(
            {
                "tool": "ultrasound_reference_search",
                "status": "features_only",
                "query_image": query,
                "p_he": None,
                "neighbors": neighbors,
                "evidence": "Frozen TinyUSFM cosine similarity; no HE-trained classification head.",
                "limitations": [
                    "Similarity is not disease probability",
                    "Reference-pool-only demo",
                    "Patient/near-duplicate exclusion beyond self is not implemented",
                ],
            }
        )
    unchanged = all(file_sha256(data_root / name) == value for name, value in before.items())
    if not unchanged:
        raise RuntimeError("Source image changed during inference")
    summary = {
        "status": "engineering_smoke_test_passed",
        "clinical_validation": False,
        "encoder": "TinyUSFM CLS token",
        "feature_shape": list(features.shape),
        "parameters": sum(p.numel() for p in encoder.model.parameters()),
        "checkpoint_sha256": OFFICIAL_SHA256,
        "split_sha256": file_sha256(split_path),
        "reference_counts": dict(Counter(str(v) for v in labels.values())),
        "test_images_used": 0,
        "test_labels_used_for_fitting_or_retrieval": False,
        "trainable_parameters": 0,
        "p_he_available": False,
        "elapsed_seconds_including_load": round(elapsed, 3),
        "source_images_unchanged": unchanged,
        "torch_version": torch.__version__,
        "device": "cpu",
        "warning": "Generic ultrasound encoder only; no validated HE classifier. See reference_counts.",
    }
    output.mkdir(parents=True)
    np.save(output / "reference_features.npy", features, allow_pickle=False)
    for filename, value in [
        ("summary.json", summary),
        ("reference_index.json", names),
        ("tool_examples.json", tools),
        ("image_fingerprints.json", before),
    ]:
        (output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    """在指定参考池上执行离线编码实验，将特征与审计结果写入新目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.project / "data/unified",
                args.project / "configs/reference_split.json",
                args.checkpoint,
                args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
