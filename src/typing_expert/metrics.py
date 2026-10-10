"""Fixed-denominator classification scores and paired group comparisons."""

import math

import numpy as np

from . import CLASSES


def score(rows, predictions, breakdown=True):
    planned = {r["image"]: r for r in rows}
    observed = {r["image"]: r for r in predictions}
    if (
        len(planned) != len(rows)
        or len(observed) != len(predictions)
        or not set(observed) <= set(planned)
    ):
        raise ValueError("Duplicate or unexpected predictions")
    columns = (*CLASSES, "CL", "error", "abstain", "unattempted")
    matrix = {c: dict.fromkeys(columns, 0) for c in CLASSES}
    bins = [dict(count=0, confidence_sum=0.0, correct=0) for _ in range(5)]
    nll, brier, probabilistic = 0.0, 0.0, 0
    for name, row in planned.items():
        pred = observed.get(name, {"status": "unattempted"})
        label = pred["predicted_type"] if pred["status"] == "ok" else pred["status"]
        matrix[row["label"]][label] += 1
        if pred["status"] == "ok":
            confidence = pred["confidence"]
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence")
            bucket = bins[min(4, int(confidence * 5))]
            bucket["count"] += 1
            bucket["confidence_sum"] += confidence
            bucket["correct"] += int(label == row["label"])
            if "probabilities" in pred:
                p = np.array(pred["probabilities"], dtype=float)
                if (
                    p.shape != (7,)
                    or not np.isfinite(p).all()
                    or (p < 0).any()
                    or not np.isclose(p.sum(), 1)
                ):
                    raise ValueError("Invalid seven-class probabilities")
                y = CLASSES.index(row["label"])
                nll -= math.log(max(float(p[y]), 1e-15))
                brier += float(((p - np.eye(7)[y]) ** 2).sum())
                probabilistic += 1
    per_class = {}
    for c in CLASSES:
        support = sum(matrix[c].values())
        tp = matrix[c][c]
        predicted = sum(matrix[t][c] for t in CLASSES)
        per_class[c] = dict(
            count=support,
            correct=tp,
            recall=tp / support if support else None,
            precision=tp / predicted if predicted else None,
            f1=2 * tp / (support + predicted) if support else None,
        )
    present = [v for v in per_class.values() if v["count"]]
    total = len(rows)
    correct = sum(matrix[c][c] for c in CLASSES)
    valid = sum(matrix[c][p] for c in CLASSES for p in (*CLASSES, "CL"))
    output = dict(
        planned=total,
        attempted=len(predictions),
        correct=correct,
        accuracy=correct / total if total else None,
        valid_predictions=valid,
        errors=sum(matrix[c]["error"] for c in CLASSES),
        unattempted=total - len(predictions),
        coverage=valid / total if total else None,
        complete=len(predictions) == total,
        confusion_matrix=matrix,
        per_class=per_class,
        macro_recall=sum(v["recall"] for v in present) / len(present) if present else None,
        macro_f1=sum(v["f1"] for v in present) / len(present) if present else None,
        probability_count=probabilistic,
        nll_valid=nll / probabilistic if probabilistic else None,
        brier_valid=brier / probabilistic if probabilistic else None,
        confidence_bins=[
            dict(
                lower=i / 5,
                upper=(i + 1) / 5,
                count=b["count"],
                mean_confidence=b["confidence_sum"] / b["count"] if b["count"] else None,
                accuracy=b["correct"] / b["count"] if b["count"] else None,
            )
            for i, b in enumerate(bins)
        ],
    )
    if breakdown:
        selections = {
            s: [r for r in rows if r["source_dataset"] == s]
            for s in sorted({r["source_dataset"] for r in rows})
        }
        selections["without_web_normal"] = [
            r for r in rows if "web_liver_ultrasound_20260918" not in r["source_dataset"]
        ]
        selections["hospital"] = [r for r in rows if "cooperating_hospital" in r["source_dataset"]]
        output["by_source_or_subset"] = {
            k: score(
                sub, [p for p in predictions if p["image"] in {r["image"] for r in sub}], False
            )
            for k, sub in selections.items()
        }
    return output


def paired_comparison(rows, first, second, repeats=20000):
    """Group resampling; not a claim of patient independence."""
    if any(
        set(p["image"] for p in predictions) != {r["image"] for r in rows}
        for predictions in (first, second)
    ):
        raise ValueError("Comparison requires complete matching image sets")
    score(rows, first, False)
    score(rows, second, False)
    maps = [{p["image"]: p for p in predictions} for predictions in (first, second)]
    groups = {}
    for row in rows:
        outcomes = [
            int(
                m[row["image"]]["status"] == "ok"
                and m[row["image"]]["predicted_type"] == row["label"]
            )
            for m in maps
        ]
        groups.setdefault(row["group_id"], []).append(outcomes[0] - outcomes[1])
    sizes = np.array([len(x) for x in groups.values()])
    deltas = np.array([sum(x) for x in groups.values()])
    rng = np.random.default_rng(42)
    draws = rng.integers(len(groups), size=(repeats, len(groups)))
    differences = deltas[draws].sum(1) / sizes[draws].sum(1)
    return dict(
        first_minus_second=float(deltas.sum() / sizes.sum()),
        group_bootstrap_95_interval=np.quantile(differences, [0.025, 0.975]).tolist(),
        groups=len(groups),
        repeats=repeats,
        seed=42,
        limitation="Known association groups; patient independence unverified",
    )
