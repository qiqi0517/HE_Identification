"""Fixed-denominator seven-class scores and paired association-group comparisons."""

import random

from models import ModelResponse
from selective_review import token_usage
from typing_protocol import CLASSES, parse_prediction, report

from .artifacts import canonical
from .policy import ACTIONS, correct, normalize, route


def validate_prediction(pred):
    if pred.get("status") == "error":
        if pred.get("predicted_type") is not None or pred.get("confidence") is not None:
            raise ValueError("Error cannot contain a label/confidence")
        return
    import json

    parse_prediction(
        ModelResponse(
            json.dumps(
                {k: pred[k] for k in ("status", "predicted_type", "confidence", "findings")}
            ),
            True,
        )
    )


def validate_bank(bank):
    if bank.get("schema") != "he-typing-action-bank-v1" or bank["repetitions"] != 3:
        raise ValueError("Incompatible typing action bank")
    seen = set()
    for sample in bank["samples"]:
        if sample["image"] in seen or sample["label"] not in CLASSES or len(sample["repeats"]) != 3:
            raise ValueError("Invalid/duplicate bank sample")
        seen.add(sample["image"])
        for rep in sample["repeats"]:
            validate_prediction(rep["initial"])
            expected = (
                {"initial", *ACTIONS[1:]} if rep["initial"]["status"] == "ok" else {"initial"}
            )
            if set(rep) != expected:
                raise ValueError("Incomplete or unexpected action results")
            for pred in rep.values():
                validate_prediction(pred)
    if not seen:
        raise ValueError("Empty bank")


def replay(bank, policy):
    validate_bank(bank)
    policy = normalize(policy)
    rows = []
    for sample in bank["samples"]:
        for rep, obs in enumerate(sample["repeats"]):
            action = route(policy, obs["initial"])
            rows.append(
                dict(
                    image=sample["image"],
                    rep=rep,
                    initial=obs["initial"],
                    action=action,
                    final=obs["initial"] if action == "ACCEPT" else obs[action],
                )
            )
    return rows


def evaluate(rows, samples):
    expected = {(s["image"], r) for s in samples for r in range(3)}
    if len(rows) != len(expected) or {(r["image"], r["rep"]) for r in rows} != expected:
        raise ValueError("Evaluation requires every planned image in all three rounds")
    lookup = {s["image"]: s for s in samples}
    repeats, total_correct, calls, rescued, harmed = [], 0, 0, 0, 0
    bins = [dict(count=0, confidence_sum=0.0, correct=0) for _ in range(10)]
    stages = []
    for row in rows:
        validate_prediction(row["initial"])
        validate_prediction(row["final"])
        if row["action"] not in ACTIONS or (
            row["initial"]["status"] != "ok" and row["action"] != "ACCEPT"
        ):
            raise ValueError("Invalid executed route")
        if row["action"] == "ACCEPT" and row["final"] != row["initial"]:
            raise ValueError("ACCEPT must preserve initial response")
        label = lookup[row["image"]]["label"]
        before, after = correct(row["initial"], label), correct(row["final"], label)
        reviewed = row["action"] != "ACCEPT"
        calls += 1 + reviewed
        total_correct += after
        rescued += reviewed and not before and after
        harmed += reviewed and before and not after
        stages.append(row["initial"])
        if reviewed:
            stages.append(row["final"])
        if row["initial"]["status"] == "ok":
            q = row["initial"]["confidence"]
            b = bins[min(9, int(q * 10))]
            b["count"] += 1
            b["confidence_sum"] += q
            b["correct"] += before
    for rep in range(3):
        records = [lookup[r["image"]] for r in rows if r["rep"] == rep]
        finals = [r["final"] for r in rows if r["rep"] == rep]
        records = [{**meta, **pred} for meta, pred in zip(records, finals)]
        repeats.append(report(records, samples))
    return dict(
        correct=total_correct,
        planned=len(rows),
        accuracy=total_correct / len(rows),
        calls=calls,
        reviews=calls - len(rows),
        rescued=rescued,
        harmed=harmed,
        repeats=repeats,
        usage=token_usage(stages),
        latency_seconds=sum(s.get("latency_seconds", 0) for s in stages),
        confidence_bins=[
            dict(
                lower=i / 10,
                upper=(i + 1) / 10,
                count=b["count"],
                mean_confidence=b["confidence_sum"] / b["count"] if b["count"] else None,
                accuracy=b["correct"] / b["count"] if b["count"] else None,
            )
            for i, b in enumerate(bins)
        ],
        cost_note="Returned token usage only; excludes unreported retries. Monetary cost unavailable without pricing.",
    )


def rank_key(candidate):
    score = candidate["score"]
    return (
        -score["correct"],
        score["calls"],
        canonical(candidate["spec"]),
        canonical(candidate["policy"]),
    )


def paired(first, second, samples, repetitions=20000):
    lookup = {s["image"]: s for s in samples}
    fm = {(r["image"], r["rep"]): r for r in first}
    sm = {(r["image"], r["rep"]): r for r in second}
    expected = {(n, r) for n in lookup for r in range(3)}
    if set(fm) != expected or set(sm) != expected:
        raise ValueError("Paired comparison requires matching complete rows")
    groups = {}
    for n, s in lookup.items():
        delta = (
            sum(
                correct(fm[n, r]["final"], s["label"]) - correct(sm[n, r]["final"], s["label"])
                for r in range(3)
            )
            / 3
        )
        group = groups.setdefault(s["group_id"], [0.0, 0])
        group[0] += delta
        group[1] += 1
    values, rng = list(groups.values()), random.Random(42)
    draws = []
    for _ in range(repetitions):
        selected = rng.choices(values, k=len(values))
        draws.append(sum(v[0] for v in selected) / sum(v[1] for v in selected))
    draws.sort()
    return dict(
        first_minus_second=sum(v[0] for v in values) / len(samples),
        group_bootstrap_95_interval=[
            draws[int(0.025 * repetitions)],
            draws[int(0.975 * repetitions)],
        ],
        groups=len(values),
        repetitions=repetitions,
        seed=42,
        limitation="Known association groups, not verified independent patients; validation-adaptive selection.",
    )
