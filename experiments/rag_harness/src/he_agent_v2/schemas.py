"""推理边界与严格观察协议：查询对象没有诊断标签字段。"""

import json
from dataclasses import dataclass
from pathlib import Path

from he_agent import parse_prediction
from models import ModelResponse

TAGS = ("cystic", "mixed", "solid", "calcified", "no_focal_lesion", "uncertain")
QUALITIES = {"good", "limited", "uninterpretable"}


@dataclass(frozen=True)
class Query:
    """从本地评价样本显式剥离真值和来源，只携带图像及防泄漏组信息。"""

    image: str
    path: Path
    mime: str
    sha256: str
    group_id: str
    group_keys: tuple[str, ...]

    @classmethod
    def from_sample(cls, sample, dataset, split):
        """从本地样本和划分生成查询对象，保留组身份但不携带诊断真值。"""
        meta = split["samples"][sample["image"]]
        return cls(
            sample["image"],
            dataset / sample["image"],
            sample["mime"],
            sample["sha256"],
            meta["group_id"],
            tuple(meta.get("provenance_group_keys", [])),
        )


def parse_observation(response):
    """观察字段不合法时整份观察失败；概率协议与 v1 完全一致。"""
    if not response.completed or not isinstance(response.text, str):
        raise ValueError("Incomplete observation")

    def unique(pairs):
        """拒绝观察结果中的重复 JSON 键，避免冲突征象被静默覆盖。"""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate observation key")
            result[key] = value
        return result

    text = response.text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    obj = json.loads(text, object_pairs_hook=unique)
    fields = {"quality", "appearance_tags", "observations", "limitations", "p_he", "evidence"}
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError("Unexpected observation fields")
    if not isinstance(obj["quality"], str) or obj["quality"] not in QUALITIES:
        raise ValueError("Unknown quality")
    tags = obj["appearance_tags"]
    if (
        not isinstance(tags, list)
        or not 1 <= len(tags) <= len(TAGS)
        or any(not isinstance(x, str) or x not in TAGS for x in tags)
        or len(set(tags)) != len(tags)
    ):
        raise ValueError("Invalid appearance tags")
    for key in ("observations", "limitations"):
        values = obj[key]
        if (
            not isinstance(values, list)
            or len(values) > 4
            or any(not isinstance(x, str) or not x.strip() or len(x) > 200 for x in values)
        ):
            raise ValueError("Invalid concise observations")
    if not obj["observations"]:
        raise ValueError("Missing observations")
    parse_prediction(ModelResponse(json.dumps({k: obj[k] for k in ("p_he", "evidence")}), True))
    if len(obj["evidence"]) > 800:
        raise ValueError("Evidence too long")
    return obj
