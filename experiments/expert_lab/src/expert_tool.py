"""可独立调用的参考图工具：输入一张图片，返回参考信息和明确的能力边界。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from experiments.expert_lab.src.run_experiment import validated_reference_pool
from experiments.expert_lab.src.ultrasound_encoder import (
    UltrasoundEncoder,
    file_sha256,
    nearest_references,
)


def query_references(image: Path, project: Path, checkpoint: Path, k: int = 4) -> dict:
    """不读取查询图片的真值；仅使用参考池的标签。"""
    torch.set_num_threads(2)
    data_root = project / "data/unified"
    names, labels, _ = validated_reference_pool(data_root, project / "configs/reference_split.json")
    query_hash = file_sha256(image)
    # 即使查询图片被复制/改名，也排除字节相同的参考图。
    names = [name for name in names if file_sha256(data_root / name) != query_hash]
    if not names:
        raise ValueError("No eligible reference images remain")
    encoder = UltrasoundEncoder(checkpoint)
    features = np.stack([encoder.encode(data_root / name) for name in names])
    neighbors = nearest_references(encoder.encode(image), features, names, "", k)
    for item in neighbors:
        item["reference_label"] = labels[item["image"]]
    return {
        "tool": "ultrasound_reference_search",
        "status": "features_only",
        "p_he": None,
        "query_sha256": query_hash,
        "neighbors": neighbors,
        "evidence": "Frozen ultrasound encoder retrieves labeled reference images.",
        "limitations": [
            "Similarity is not HE probability",
            "No HE-trained classification head",
            "Near-duplicate/patient filtering needs further implementation",
        ],
    }


def main() -> None:
    """读取查询图片、项目和权重路径，输出相似参考及工具能力边界，不生成疾病概率。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            query_references(args.image, args.project, args.checkpoint, args.k),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
