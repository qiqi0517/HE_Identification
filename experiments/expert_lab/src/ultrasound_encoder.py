"""TinyUSFM 的只读特征工具；不把通用超声特征解释为患病概率。"""

from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms

OFFICIAL_SHA256 = "776a50eeff402802c8cd06e00fe9de73df4161504fa733284bc240904b48252c"


def file_sha256(path: Path) -> str:
    """对文件字节计算指纹，不改变原图。"""
    return sha256(path.read_bytes()).hexdigest()


def safe_state_dict(path: Path) -> dict:
    """仅加载张量权重；不回退到可以执行任意 pickle 的加载模式。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a non-empty state_dict")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("State dict must contain only named tensors")
    return state


def build_encoder() -> VisionTransformer:
    """按作者的 192 维、12 层、12 头配置构造模型，保留预训练 CLS/norm。"""
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        global_pool="token",
    )
    # 显式采用常规 attention，方便与作者代码做数值一致性检查。
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


class UltrasoundEncoder:
    """冻结的通用超声编码器，输出 192 维向量，不含疾病分类头。"""

    def __init__(self, checkpoint: Path):
        """核对权重哈希并清理键名前缀，严格加载编码器后冻结参数以便只读推理。"""
        if file_sha256(checkpoint) != OFFICIAL_SHA256:
            raise ValueError("Checkpoint SHA-256 differs from the verified public release")
        state = safe_state_dict(checkpoint)
        cleaned = {}
        for key, value in state.items():
            name = key.removeprefix("module.").removeprefix("model.")
            if name in cleaned:
                raise ValueError("Duplicate normalized checkpoint key")
            cleaned[name] = value
        self.model = build_encoder()
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval().requires_grad_(False)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def tensor(self, image_path: Path) -> torch.Tensor:
        """按作者验证预处理在内存缩放；原文件保持原分辨率和原字节。"""
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")).unsqueeze(0)

    def encode(self, image_path: Path) -> np.ndarray:
        """提取单图 192 维特征并做 L2 归一化；返回有限 float32 向量用于余弦检索。"""
        with torch.inference_mode():
            feature = self.model(self.tensor(image_path))[0].cpu().numpy()
        if feature.shape != (192,) or not np.isfinite(feature).all():
            raise ValueError("Invalid encoder feature")
        norm = np.linalg.norm(feature)
        if norm <= 0:
            raise ValueError("Zero feature norm")
        return (feature / norm).astype(np.float32)


def nearest_references(
    query: np.ndarray, features: np.ndarray, names: list[str], query_name: str, k: int = 4
) -> list[dict]:
    """仅在调用方已校验的参考池检索，并排除查询图本身。"""
    if k < 1 or features.shape != (len(names), 192) or query.shape != (192,):
        raise ValueError("Invalid retrieval inputs")
    if not np.isfinite(features).all() or not np.isfinite(query).all():
        raise ValueError("Non-finite retrieval features")
    scores = features @ query
    eligible = [i for i, name in enumerate(names) if name != query_name]
    ordered = sorted(eligible, key=lambda i: (-float(scores[i]), names[i]))[:k]
    return [{"image": names[i], "cosine_similarity": float(scores[i])} for i in ordered]
