"""Training-only transforms and independently configurable stopping criteria."""

import math
from collections import Counter

import torch
from torch import nn

from . import CLASSES


class Standardize(nn.Module):
    """Statistics are fitted on training features and saved inside the checkpoint."""

    def __init__(self, dimension):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))

    def fit(self, features):
        if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
            raise ValueError("Invalid training features for standardization")
        values = features.double()
        mean, std = values.mean(0), values.std(0, unbiased=False)
        self.mean.copy_(mean.to(self.mean))
        self.scale.copy_(torch.where(std < 1e-6, torch.ones_like(std), std).to(self.scale))

    def forward(self, features):
        return (features - self.mean) / self.scale


def configure_head(model, scaling):
    if scaling == "standard":
        model.head = nn.Sequential(Standardize(model.dimension), model.head)
    elif scaling != "none":
        raise ValueError("Unknown feature scaling")


def class_weights(rows, mode):
    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError("Unknown class weighting")
    counts = Counter(row["label"] for row in rows)
    if any(counts[c] == 0 for c in CLASSES):
        raise ValueError("Class weighting requires every class in training")
    weights = torch.tensor([1 / math.sqrt(counts[c]) for c in CLASSES])
    return weights / weights.mean()


def resolve_options(args):
    defaults = dict(
        feature_scaling="none",
        weight_decay=1e-4,
        head_lr=1e-3,
        finetune_head_lr=1e-4,
        encoder_lr=1e-5,
        early_stop_metric="accuracy",
        patience=8 if args.variant == "S3" else 10,
        class_weight="none",
    )
    values = {
        k: getattr(args, k, None) if getattr(args, k, None) is not None else v
        for k, v in defaults.items()
    }
    for key in ("weight_decay", "head_lr", "finetune_head_lr", "encoder_lr"):
        value = values[key]
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"Invalid {key}")
    if values["patience"] < 1:
        raise ValueError("Patience must be positive")
    if args.variant == "S3" and values["feature_scaling"] != "none":
        raise ValueError("Feature standardization is supported only for frozen S1/S2 encoders")
    if values["early_stop_metric"] not in ("accuracy", "loss"):
        raise ValueError("Invalid early stopping metric")
    return values, values == defaults


def stopping_key(correct, loss, metric):
    return (correct, -loss) if metric == "accuracy" else (-loss,)
