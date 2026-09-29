"""仅在推理结束后接入真值：概率评分、成对纠正/退步和调用成本。"""

from collections import Counter

from he_agent import metrics, report


def evaluate(results, samples, threshold=0.5):
    """排除 dry-run，复用主指标并在两阶段均有效的病例上统计纠正和退化。"""
    scored = [r for r in results if r["status"] != "dry_run"]
    output = report(scored, samples, threshold)
    pairs = [
        r
        for r in scored
        if r.get("stage1") and r["stage1"]["status"] == "ok" and r["status"] == "ok"
    ]
    first = [dict(r, p_he=r["stage1"]["prediction"]["p_he"]) for r in pairs]
    changes = Counter()
    by_class = {str(y): {"n": 0, "corrections": 0, "regressions": 0} for y in (0, 1)}
    for r in pairs:
        y = r["true_label"]
        before = int(r["stage1"]["prediction"]["p_he"] >= threshold) == y
        after = int(r["p_he"] >= threshold) == y
        key = (
            "corrections"
            if not before and after
            else "regressions"
            if before and not after
            else "unchanged"
        )
        changes[key] += 1
        by_class[str(y)]["n"] += 1
        if key != "unchanged":
            by_class[str(y)][key] += 1
    output["paired_stage1_stage2"] = {
        "n": len(pairs),
        "stage1": metrics(first, len(pairs), threshold),
        "stage2": metrics(pairs, len(pairs), threshold),
        "changes": dict(changes),
        "by_true_class": by_class,
    }
    calls = [c for r in results for c in r["calls"]]
    tokens = Counter()
    for call in calls:
        for key, value in (call["usage"] or {}).items():
            if type(value) is int:
                tokens[key] += value
    output["cost"] = {
        "logical_calls": len(calls),
        "http_attempts": sum(c["http_attempts"] for c in calls),
        "latency_seconds": sum(r["latency_seconds"] for r in results),
        "reported_usage": dict(tokens),
        "calls_without_usage": sum(c["usage"] is None for c in calls),
        "note": "Only returned usage is counted; failed/retried requests may incur unknown cost.",
    }
    output["stage_status_counts"] = {
        stage: dict(Counter((r.get(stage) or {}).get("status", "not_called") for r in results))
        for stage in ("stage1", "stage2")
    }
    output["finish_reason_counts"] = dict(
        Counter(
            str((r.get(stage) or {}).get("response", {}).get("finish_reason"))
            for r in results
            for stage in ("stage1", "stage2")
            if (r.get(stage) or {}).get("response")
        )
    )
    output["error_code_counts"] = dict(Counter(r["error"] for r in results if r.get("error")))
    output["dry_run_samples"] = sum(r["status"] == "dry_run" for r in results)
    output["routing_fallbacks"] = sum(r.get("routing_fallback", False) for r in results)
    output["branch_counts"] = dict(Counter(b for r in results for b in r["branches"]))
    return output
