"""选择性复核：分流不读真值，阶段记录与配对评价在本地完成。

职责分工：needs_review 只做门控；predict_stage 只做调用与解析；
review_report/token_usage 在本地比较两阶段并汇总用量，不参与模型判断。
"""

import math
import time

from models import ModelError


def validate_review_threshold(lower, upper):
    """复核区间为开区间 (lower, upper)，upper 与辅助分类阈值一致。"""
    if type(lower) not in (int, float) or not math.isfinite(lower) or not 0 <= lower < upper:
        raise ValueError("--review-threshold must be finite and satisfy 0 <= value < --threshold")


def needs_review(prediction, lower, upper):
    """只复核有效、位于区间内的阴性概率；拒答与错误不视作阴性。"""
    return prediction["status"] == "ok" and lower < prediction["p_he"] < upper


def predict_stage(client, request, parser, threshold):
    """执行一次逻辑调用，保留响应及耗时；网络重试由原适配器负责。"""
    started = time.monotonic()
    record = dict(
        status="error",
        p_he=None,
        p_non_he=None,
        predicted_label=None,
        fatal=False,
        evidence=None,
        response_id=None,
        returned_model=None,
        usage=None,
        finish_reason=None,
        raw_answer=None,
    )
    try:
        response = client.generate(request)
        record.update(
            response_id=response.response_id,
            returned_model=response.model,
            usage=response.usage,
            finish_reason=response.finish_reason,
            raw_answer=response.text,
        )
        prediction = parser(response)
        probability = prediction["p_he"]
        record.update(
            p_he=probability,
            p_non_he=None if probability is None else 1 - probability,
            predicted_label=None if probability is None else int(probability >= threshold),
            evidence=prediction["evidence"],
            status="abstain" if probability is None else "ok",
        )
    except ModelError as error:
        # 适配器提供的 ModelError 已脱敏；不输出服务商原始异常正文。
        record.update(error=str(error), fatal=error.fatal)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        record["error"] = f"Invalid input/response ({type(error).__name__})"
    record["latency_seconds"] = round(time.monotonic() - started, 3)
    return record


def token_usage(stages):
    """统一各适配器已返回的 token 计数；缺失不当作零或真实账单。"""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    reported = dict.fromkeys(totals, 0)
    for stage in stages:
        usage = stage.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        for field, alternative in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(field, usage.get(alternative))
            if type(value) is int and value >= 0:
                totals[field] += value
                reported[field] += 1
    return {
        **{field: value if reported[field] else None for field, value in totals.items()},
        "stages_reporting": reported,
        "logical_calls": len(stages),
        "scope": "returned_usage_only_excludes_unreported_retries",
    }


def review_report(results, samples, threshold, base_report):
    """报告初判、最终和共同有效样本；不把覆盖率下降伪装成改善。"""
    # 保留本地真值与来源，再用初判字段覆盖最终预测，恢复同一 run 的初判视图。
    initial = [{**row, **(row.get("initial_prediction") or {})} for row in results]
    final_report = base_report(results, samples, threshold)
    reviewed = [row for row in results if row.get("reviewed")]
    # 必须在两阶段都有效的同一病例上比较，否则复核弃权会虚假降低 FN。
    pairs = [
        (before, after)
        for before, after in zip(initial, results)
        if before["status"] == after["status"] == "ok"
    ]
    pair_images = {after["image"] for _, after in pairs}
    pair_samples = [sample for sample in samples if sample["image"] in pair_images]
    changes = {
        "rescued_fn": 0,
        "introduced_fp": 0,
        "probability_up": 0,
        "probability_down": 0,
        "probability_unchanged": 0,
    }
    for before, after in pairs:
        if not after.get("reviewed"):
            continue
        # 在线只复核初判阴性，所以跨过阈值后要按本地真值区分救回与误报。
        if before["p_he"] < threshold <= after["p_he"]:
            key = "rescued_fn" if after["true_label"] == 1 else "introduced_fp"
            changes[key] += 1
        delta = after["p_he"] - before["p_he"]
        changes[
            "probability_up"
            if delta > 0
            else "probability_down"
            if delta < 0
            else "probability_unchanged"
        ] += 1
    stages = [
        stage
        for row in results
        for stage in (row.get("initial_prediction"), row.get("review_prediction"))
        if stage is not None
    ]
    final_report["selective_review"] = {
        "initial": base_report(initial, samples, threshold),
        "paired_valid": {
            "count": len(pairs),
            "initial": base_report([before for before, _ in pairs], pair_samples, threshold),
            "final": base_report([after for _, after in pairs], pair_samples, threshold),
        },
        "reviewed": len(reviewed),
        "review_rate_over_attempted": len(reviewed) / len(results) if results else None,
        "review_valid": sum(row["status"] == "ok" for row in reviewed),
        "review_abstentions": sum(row["status"] == "abstain" for row in reviewed),
        "review_errors": sum(row["status"] == "error" for row in reviewed),
        "changes_on_paired_reviewed": changes,
        "usage_total": token_usage(stages),
        "review_latency_seconds": round(
            sum(row["review_prediction"]["latency_seconds"] for row in reviewed), 3
        ),
        "stage_latency_seconds": round(sum(stage["latency_seconds"] for stage in stages), 3),
    }
    return final_report
