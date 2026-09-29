"""固定参考原图：复用 v1 选择顺序，不依赖编码器、向量或索引。"""

import hashlib
import json

from PIL import Image

from reference_split import select_references


class StaleData(ValueError):
    """运行中图像变化时停止；不使用验证之前缓存的旧内容掩盖变化。"""


class UnsafeReferences(ValueError):
    """固定示例与查询同组或近重复，不替换示例继续实验。"""


def sha(data):
    """计算原始字节 SHA-256，用于判断参考文件是否变化。"""
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    """将元数据按键排序、紧凑编码为 UTF-8 字节，得到稳定的指纹输入。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def fingerprint(path):
    """仅在内存计算解码像素和dHash，上传仍使用原始字节。"""
    data = path.read_bytes()
    with Image.open(path) as source:
        image = source.convert("RGB")
        pixel = sha(canonical(image.size) + image.tobytes())
        values = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    bits = sum(
        (values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8)
        for x in range(8)
    )
    return {"sha256": sha(data), "pixel_sha256": pixel, "dhash": f"{bits:016x}"}


class FixedReferences:
    def __init__(self, dataset, samples, split, count=8):
        """复用主流程参考选择，逐图核对文件指纹，缓存可审计的固定参考记录。"""
        self.dataset = dataset
        self.rows = []
        # 与主 Agent 的 reference-count=4/6/8 使用相同图片和顺序。
        for sample in select_references(split, samples, count):
            name = sample["image"]
            meta = split["samples"][name]
            fp = fingerprint(dataset / name)
            if fp["sha256"] != sample["sha256"]:
                raise StaleData("Reference changed since validation")
            self.rows.append(
                {
                    "image": name,
                    "label": sample["label"],
                    "mime": sample["mime"],
                    "source_dataset": sample["source_dataset"],
                    "group_id": meta["group_id"],
                    "group_keys": meta.get("provenance_group_keys", []),
                    **fp,
                }
            )
        self.digest = sha(canonical(self.rows))

    def prepare(self, query):
        """校验固定示例与查询隔离；同一实验中不按查询切换或删减参考图。"""
        if sha(query.path.read_bytes()) != query.sha256:
            raise StaleData("Query changed since validation")
        fp = fingerprint(query.path)
        refs = []
        for row in self.rows:
            data = (self.dataset / row["image"]).read_bytes()
            if sha(data) != row["sha256"]:
                raise StaleData("Reference changed during run")
            if (
                row["image"] == query.image
                or row["sha256"] == query.sha256
                or row["pixel_sha256"] == fp["pixel_sha256"]
                or row["group_id"] == query.group_id
                or set(row["group_keys"]) & set(query.group_keys)
                or (int(row["dhash"], 16) ^ int(fp["dhash"], 16)).bit_count() <= 4
            ):
                raise UnsafeReferences("Fixed reference overlaps query; audit split")
            refs.append({"bytes": data, "mime": row["mime"], "label": row["label"]})
        return refs, {
            "strategy": "fixed",
            "count": len(refs),
            "references_sha256": self.digest,
            "references": self.rows,
        }
