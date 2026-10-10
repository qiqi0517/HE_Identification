"""分型协议、请求和固定分母评分；待判图标签只用于本地评分。"""

import json
import math
from collections import Counter

from models import ImagePart, ModelRequest, TextPart

CLASSES = ("NO_HE", "CE1", "CE2", "CE3", "CE4", "CE5", "AE")
OUTPUT_TYPES = CLASSES
SCHEMA = "he-typing-confidence-v1"


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def parse_prediction(response):
    if not response.completed or not isinstance(response.text, str):
        raise ValueError("Response incomplete or missing")
    text = response.text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    obj = json.loads(text, object_pairs_hook=unique_fields)
    if not isinstance(obj, dict) or set(obj) != {
        "status",
        "predicted_type",
        "confidence",
        "findings",
    }:
        raise ValueError("Expected status, predicted_type, confidence and findings")
    if obj["status"] not in ("ok", "abstain"):
        raise ValueError("Invalid prediction status")
    findings = obj["findings"]
    if (
        not isinstance(findings, list)
        or not 1 <= len(findings) <= 5
        or any(not isinstance(t, str) or not t.strip() or len(t) > 1000 for t in findings)
    ):
        raise ValueError("Expected one to five brief findings")
    if obj["status"] == "abstain":
        if obj["predicted_type"] is not None or obj["confidence"] is not None:
            raise ValueError("Abstention must use null type and confidence")
    else:
        if obj["predicted_type"] not in OUTPUT_TYPES:
            raise ValueError("Unknown type")
        c = obj["confidence"]
        if type(c) not in (int, float) or not math.isfinite(c) or not 0 <= c <= 1:
            raise ValueError("Confidence must be finite in [0, 1]")
    return obj


def build_request(image_bytes, mime, prompt, references=()):
    parts = []
    if references:
        parts.append(TextPart("以下为已标注参考图，只作为类别示例；最后一张是唯一待判断图。"))
        for i, reference in enumerate(references, 1):
            if reference["label"] not in CLASSES:
                raise ValueError("Reference must have a typing label")
            parts += [
                TextPart(f"参考图 {i}，分型标签：{reference['label']}。"),
                ImagePart(data=reference["bytes"], mime=reference["mime"]),
            ]
    parts += [
        TextPart("请仅判断最后这张超声图的分型并给出 confidence，按指定 JSON 返回。"),
        ImagePart(data=image_bytes, mime=mime),
    ]
    return ModelRequest(system_prompt=prompt, content=tuple(parts))


def metrics(results, samples):
    """Failures, abstentions and unattempted cases remain in the planned denominator."""
    planned = {s["image"]: s for s in samples}
    observed = {r["image"]: r for r in results}
    if len(observed) != len(results) or not set(observed) <= set(planned):
        raise ValueError("Duplicate/unplanned result")
    columns = (*OUTPUT_TYPES, "abstain", "error", "unattempted")
    confusion = {label: dict.fromkeys(columns, 0) for label in CLASSES}
    correct, valid = 0, 0
    statuses = Counter()
    for image, sample in planned.items():
        row = observed.get(image)
        status = row["status"] if row else "unattempted"
        statuses[status] += 1
        prediction = row["predicted_type"] if status == "ok" else status
        confusion[sample["label"]][prediction] += 1
        valid += int(status == "ok")
        correct += int(status == "ok" and prediction == sample["label"])
    recall = {}
    for label, cells in confusion.items():
        n = sum(cells.values())
        recall[label] = {
            "correct": cells[label],
            "count": n,
            "recall": cells[label] / n if n else None,
        }
    defined = [v["recall"] for v in recall.values() if v["recall"] is not None]
    total = len(samples)
    return {
        "output_schema": SCHEMA,
        "planned": total,
        "attempted": len(results),
        "correct": correct,
        "accuracy": correct / total if total else None,
        "valid_predictions": valid,
        "coverage": valid / total if total else None,
        "abstentions": statuses["abstain"],
        "errors": statuses["error"],
        "unattempted": statuses["unattempted"],
        "confusion_matrix": confusion,
        "per_class": recall,
        "macro_recall_present_classes": sum(defined) / len(defined) if defined else None,
        "confidence_status": "uncalibrated_self_assessment_of_label_correctness",
    }


def report(results, samples):
    output = metrics(results, samples)
    output["by_source"] = {
        source: metrics(
            [r for r in results if r["source_dataset"] == source],
            [s for s in samples if s["source_dataset"] == source],
        )
        for source in sorted({s["source_dataset"] for s in samples})
    }
    return output
