"""Versioned preprocessing and strictly loaded pretrained encoders."""

import importlib.util
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .data import PROJECT, sha

PREPROCESS = dict(
    version="rgb-resize224-imagenet-v1",
    size=[224, 224],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    interpolation="bilinear",
    antialias=True,
    features="raw_not_l2_normalized",
)
TINY_PATH = PROJECT / "experiments/expert_lab/weights/TinyUSFM.pth"


def tiny_module():
    spec = importlib.util.spec_from_file_location(
        "he_verified_ultrasound_encoder",
        PROJECT / "experiments/expert_lab/src/ultrasound_encoder.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device_for(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def transform(augment=False):
    ops = []
    if augment:
        ops += [
            transforms.RandomAffine(
                5, translate=(0.03, 0.03), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1)),
        ]
    return transforms.Compose(
        ops
        + [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
        ]
    )


def tensor(path, augment=False):
    with Image.open(path) as image:
        return transform(augment)(image.convert("RGB"))


class Expert(nn.Module):
    def __init__(self, variant, pretrained=False, tiny_weights=TINY_PATH):
        super().__init__()
        if variant not in ("S1", "S2", "S3"):
            raise ValueError("Unknown expert variant")
        self.variant = variant
        self.pretrained_hash = None
        if variant in ("S1", "S3"):
            module = tiny_module()
            self.encoder = (
                module.UltrasoundEncoder(Path(tiny_weights)).model
                if pretrained
                else module.build_encoder()
            )
            self.dimension = 192
            if pretrained:
                self.pretrained_hash = sha(tiny_weights)
        else:
            self.encoder = models.resnet18(weights=None)
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                state = weights.get_state_dict(progress=True, check_hash=True)
                self.encoder.load_state_dict(state, strict=True)
                cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
                self.pretrained_hash = sha(cached)
            self.encoder.fc = nn.Identity()
            self.dimension = 512
        self.encoder.eval().requires_grad_(False)
        self.head = nn.Linear(self.dimension, 7)

    def forward(self, images):
        return self.head(self.encoder(images))

    def phase(self, finetune=False):
        self.encoder.eval().requires_grad_(False)
        self.head.train().requires_grad_(True)
        if finetune:
            if self.variant != "S3":
                raise ValueError("Only S3 supports finetuning")
            for block in self.encoder.blocks[-2:]:
                block.train().requires_grad_(True)
            self.encoder.norm.train().requires_grad_(True)
        groups = [{"params": self.head.parameters(), "lr": 1e-4 if finetune else 1e-3}]
        if finetune:
            groups.append(
                {"params": [p for p in self.encoder.parameters() if p.requires_grad], "lr": 1e-5}
            )
        return groups
