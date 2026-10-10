"""Exact per-predicted-class, one-threshold routing; no model calls."""

import math

from typing_protocol import CLASSES

from .artifacts import canonical

ACTIONS = (
    "ACCEPT",
    "REVIEW_CHECKLIST",
    "REVIEW_REFERENCE",
    "REVIEW_KNOWLEDGE",
    "REVIEW_REFERENCE_CHECKLIST",
)
THRESHOLDS = tuple(i / 10 for i in range(1, 10))


def rule(threshold=None, low="ACCEPT", high=None):
    high = low if high is None else high
    if low not in ACTIONS or high not in ACTIONS:
        raise ValueError("Unknown typing action")
    if threshold is not None and (
        type(threshold) not in (int, float) or threshold not in THRESHOLDS
    ):
        raise ValueError("Threshold must be on the 0.1..0.9 grid")
    if threshold is None and low != high:
        raise ValueError("Unsplit rule must have equal actions")
    return dict(threshold=None if low == high else threshold, low=low, high=high)


def normalize(policy):
    if not isinstance(policy, dict) or set(policy) != set(CLASSES):
        raise ValueError("Policy must contain exactly seven predicted classes")
    out = {}
    for c in CLASSES:
        r = policy[c]
        if not isinstance(r, dict) or set(r) != {"threshold", "low", "high"}:
            raise ValueError("Invalid rule fields")
        out[c] = rule(**r)
    return out


BASELINE = {c: rule() for c in CLASSES}


def route(policy, prediction):
    if prediction["status"] != "ok":
        return "ACCEPT"
    c, q = prediction["predicted_type"], prediction["confidence"]
    if c not in CLASSES or type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Invalid initial prediction")
    r = policy[c]
    return r["low"] if r["threshold"] is None or q < r["threshold"] else r["high"]


def rules(split=True):
    result = [rule(low=a) for a in ACTIONS]
    if split:
        result += [rule(t, a, b) for t in THRESHOLDS for a in ACTIONS for b in ACTIONS if a != b]
    return sorted(result, key=canonical)


def correct(prediction, label):
    return int(prediction["status"] == "ok" and prediction["predicted_type"] == label)


def optimize(bank, split=True):
    policy, scores = {}, {}
    for c in CLASSES:
        observations = [
            (s["label"], r)
            for s in bank["samples"]
            for r in s["repeats"]
            if r["initial"]["status"] == "ok" and r["initial"]["predicted_type"] == c
        ]
        scores[c] = []
        for candidate in rules(split):
            hits, calls = 0, 0
            for label, repeat in observations:
                action = route({c: candidate}, repeat["initial"])
                hits += correct(repeat["initial"] if action == "ACCEPT" else repeat[action], label)
                calls += 1 + int(action != "ACCEPT")
            scores[c].append(dict(rule=candidate, correct=hits, calls=calls))
        policy[c] = (
            min(scores[c], key=lambda x: (-x["correct"], x["calls"], canonical(x["rule"])))["rule"]
            if observations
            else rule()
        )
    return normalize(policy), scores
