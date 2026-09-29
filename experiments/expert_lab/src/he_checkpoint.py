"""HE_DCNN 五分类权重的兼容性探针；没有疾病权重时不产生随机概率。"""

from pathlib import Path

import torch
from experiments.expert_lab.src.ultrasound_encoder import safe_state_dict
from torchvision.models import resnet50


def map_mmcls_keys(state: dict) -> dict:
    """兼容原 mmclassification 的 backbone.* 与 head.fc.* 命名。"""
    mapped = {}
    for key, value in state.items():
        name = key.removeprefix("module.")
        if name.startswith("backbone."):
            name = name.removeprefix("backbone.")
        elif name.startswith("head.fc."):
            name = "fc." + name.removeprefix("head.fc.")
        if name in mapped:
            raise ValueError("Duplicate mapped checkpoint key")
        mapped[name] = value
    return mapped


def load_he_candidate(checkpoint: Path):
    """严格验证 5 类头及所有权重形状；通过仍不等于作者临床模型复现。"""
    state = map_mmcls_keys(safe_state_dict(checkpoint))
    if "fc.weight" not in state or tuple(state["fc.weight"].shape) != (5, 2048):
        raise ValueError(
            "Expected HE five-class head; ImageNet 1000-class weights are not accepted"
        )
    model = resnet50(weights=None, num_classes=5)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def binary_probability(
    logits: torch.Tensor, positive_indices: tuple[int, ...], *, class_mapping_confirmed: bool
) -> torch.Tensor:
    """对已确认的阳性类别求概率和，禁止猜测 checkpoint 的类别顺序。"""
    if not class_mapping_confirmed:
        raise ValueError("Author/checkpoint class order must be confirmed first")
    if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [batch, 5] logits")
    if (
        not positive_indices
        or len(set(positive_indices)) != len(positive_indices)
        or any(type(i) is not int or i not in range(5) for i in positive_indices)
        or len(positive_indices) == 5
    ):
        raise ValueError("Positive indices must be a proper non-empty subset of five classes")
    return torch.softmax(logits, dim=-1)[:, list(positive_indices)].sum(dim=-1)


def checkpoint_status(checkpoint: Path | None = None) -> dict:
    """未来 Agent 工具可直接消费的未就绪状态；不影响现有 Agent。"""
    if checkpoint is None or not checkpoint.is_file():
        return {
            "tool": "he_dcnn_classifier",
            "status": "unavailable",
            "p_he": None,
            "evidence": "No verified HE-trained checkpoint available.",
        }
    load_he_candidate(checkpoint)
    return {
        "tool": "he_dcnn_classifier",
        "status": "weights_loaded_not_validated",
        "p_he": None,
        "evidence": "Class mapping, OpenCV preprocessing parity and training overlap need verification.",
    }
