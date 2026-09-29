"""冻结已有初判/复核结果，离线比较复核下界与分类阈值；不调用API。

阅读顺序：load_run 冻结快照 → materialize 按门控选阶段 → sweep 统一评分 → run 写结果。
离线分类阈值可独立于复核上界；缺少必要复核缓存的候选不能参与排名。
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from he_agent import dump, report  # noqa: E402
from project_paths import load_paths  # noqa: E402
from selective_review import needs_review, token_usage, validate_review_threshold  # noqa: E402


def finite_probability(value):
    """拒绝bool、NaN和越界值。"""
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_prediction(prediction):
    """失败/弃权保持未知，不能伪装为有效阴性。"""
    if not isinstance(prediction, dict):
        raise ValueError("Missing frozen prediction")
    status = prediction.get("status")
    probability = prediction.get("p_he")
    if status == "ok":
        if not finite_probability(probability):
            raise ValueError("Invalid frozen probability")
    elif status not in {"abstain", "error"} or probability is not None:
        raise ValueError("Invalid frozen prediction status/probability")


def load_run(path):
    """一次读取完整运行快照，在内存冻结内容并核对来源与阶段一致性。

    不读取原始图片，不跨运行混合输出，不依赖API密钥。可以分析旧单次运行，
    但它通常缺少复核缓存，所需缓存缺失的候选会被标记为不可评估。
    """
    path = path.resolve()
    blobs = {}

    def read(name):
        """读取一份快照并缓存原始字节，后续校验和复制都使用同一份内容。"""
        blobs[name] = (path / name).read_bytes()
        return blobs[name]

    config = json.loads(read("run.json"))
    if config.get("dry_run"):
        raise ValueError("Cannot sweep a dry-run: no model predictions")
    samples = json.loads(read("samples.json"))
    rows = [json.loads(line) for line in read("predictions.jsonl").splitlines() if line.strip()]
    names = [s["image"] for s in samples]
    if (
        not samples
        or len(names) != len(set(names))
        or len(samples) != config["selected"]
        or [row["image"] for row in rows] != names
    ):
        raise ValueError("Run must be complete with unique predictions in saved sample order")
    threshold = config["decision_threshold"]
    if not finite_probability(threshold):
        raise ValueError("Invalid source classification threshold")
    review = config.get("review") or {}
    enabled = review.get("enabled", False)
    if enabled:
        upper = review["upper_threshold"]
        validate_review_threshold(review["lower_threshold"], upper)
        if not finite_probability(upper) or upper != threshold:
            raise ValueError("Inconsistent source review upper threshold")
    for filename, expected in (
        ("prompt.txt", config["prompt_sha256"]),
        ("split.json", config.get("split_sha256")),
        ("review_prompt.txt", review.get("prompt_sha256") if enabled else None),
    ):
        if expected is not None:
            if hashlib.sha256(read(filename)).hexdigest() != expected:
                raise ValueError(f"Source snapshot hash mismatch: {filename}")
    if enabled and not review.get("prompt_sha256"):
        raise ValueError("Source run lacks review prompt fingerprint")
    if config.get("split_sha256"):
        split = json.loads(blobs["split.json"])
        references = json.loads(read("references.json"))
        ref_names = [r["image"] for r in references]
        if (
            len(ref_names) != config["reference_count"]
            or len(ref_names) != len(set(ref_names))
            or not set(ref_names) <= set(split["reference_pool"])
            or not set(names) <= set(split["test"])
            or set(names) & set(split["reference_pool"])
        ):
            raise ValueError("Invalid reference/test snapshot")
        for sample in samples:
            frozen = split["samples"][sample["image"]]
            if any(sample[key] != frozen[key] for key in ("label", "sha256")):
                raise ValueError("Sample/split fingerprint mismatch")
    frozen_rows = []
    for sample, row in zip(samples, rows):
        if (
            type(sample["label"]) is not int
            or sample["label"] not in (0, 1)
            or type(row["true_label"]) is not int
            or row["true_label"] != sample["label"]
            or row["source_dataset"] != sample["source_dataset"]
        ):
            raise ValueError("Prediction truth/source mismatch")
        validate_prediction(row)
        first = row.get("initial_prediction") if enabled else row
        validate_prediction(first)
        second = row.get("review_prediction") if enabled else None
        if enabled:
            eligible = needs_review(first, review["lower_threshold"], review["upper_threshold"])
            if row.get("reviewed") is not eligible or (second is not None) != eligible:
                raise ValueError("Source review gate/stage mismatch")
            if second is not None:
                validate_prediction(second)
            final = second if eligible else first
            if any(row[key] != final[key] for key in ("status", "p_he")):
                raise ValueError("Source final result does not match chosen stage")
        frozen_rows.append(
            dict(
                image=row["image"],
                true_label=row["true_label"],
                source_dataset=row["source_dataset"],
                initial=first,
                review=second,
            )
        )
    return config, samples, frozen_rows, blobs


def materialize(frozen, lower, upper, threshold):
    """只用p1分流；复核缺失返回缺失清单，不回退或只评估子集。"""
    rows, selected, missing = [], [], []
    for row in frozen:
        use_review = lower is not None and needs_review(row["initial"], lower, upper)
        stage = row["review"] if use_review else row["initial"]
        if use_review:
            selected.append(row["image"])
        if stage is None:
            missing.append(row["image"])
            continue
        probability = stage["p_he"]
        rows.append(
            dict(
                image=row["image"],
                true_label=row["true_label"],
                source_dataset=row["source_dataset"],
                p_he=probability,
                status=stage["status"],
                decision_threshold=threshold,
                predicted_label=int(probability >= threshold) if stage["status"] == "ok" else None,
                final_stage="review" if use_review else "initial",
            )
        )
    return rows, selected, missing


def paired_changes(before, after, threshold):
    """分类阈值可低于分流上界，故同时统计救回与新增的FN/FP。"""
    counts = dict(rescued_fn=0, introduced_fp=0, introduced_fn=0, corrected_fp=0)
    for a, b in zip(before, after):
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        initial_positive, final_positive = a["p_he"] >= threshold, b["p_he"] >= threshold
        if initial_positive == final_positive:
            continue
        if a["true_label"] == 1:
            key = "rescued_fn" if final_positive else "introduced_fn"
        else:
            key = "introduced_fp" if final_positive else "corrected_fp"
        counts[key] += 1
    return counts


def sweep(frozen, samples, lowers, thresholds, upper):
    """每种分类阈值自动加单次初判对照；区间开边界保持原协议。"""
    results = []
    # None 表示不开复核；每个分类阈值都必须有相同初判的单次对照。
    for lower in [None, *lowers]:
        for threshold in thresholds:
            rows, selected, missing = materialize(frozen, lower, upper, threshold)
            candidate = dict(
                id=f"case_{len(results):03d}",
                review_lower=lower,
                review_upper=upper,
                decision_threshold=threshold,
                requested_reviews=len(selected),
                missing_reviews=missing,
                status="unavailable" if missing else "ok",
            )
            # 缺任一必要缓存就不评分整组候选，防止通过忽略困难图抬高成绩。
            if missing:
                candidate.update(metrics=None, reason="required_review_predictions_missing")
            else:
                before, _, _ = materialize(frozen, None, upper, threshold)
                reviews = [r["review"] for r in frozen if r["image"] in set(selected)]
                candidate.update(
                    predictions=rows,
                    metrics=report(rows, samples, threshold),
                    changes_on_paired_valid=paired_changes(before, rows, threshold),
                    cached_review_usage=token_usage(reviews),
                    cached_review_latency_seconds=sum(r.get("latency_seconds", 0) for r in reviews),
                )
            results.append(candidate)
    available = [r for r in results if r["status"] == "ok"]
    common = set.intersection(
        *({r["image"] for r in c["predictions"] if r["status"] == "ok"} for c in available)
    )
    common_samples = [s for s in samples if s["image"] in common]
    for candidate in available:
        selected = [r for r in candidate["predictions"] if r["image"] in common]
        candidate["common_valid_metrics"] = report(
            selected, common_samples, candidate["decision_threshold"]
        )
    return results, [s["image"] for s in common_samples]


def summary_markdown(results, planned, common_count):
    """显示完整覆盖率和共同有效口径，不自动挑选测试集最优参数。"""
    lines = [
        "# 冻结预测的离线阈值回放",
        "",
        f"固定样本{planned}张；可评估方案共同有效{common_count}张。新API调用：0。",
        "复核上界与分类阈值独立；不复核行为同原单次初判。None表示该指标无定义。",
        "",
        "|复核下界|分类阈值|需要复核|有效/计划|FN|FP|loss|Brier|状态|",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]

    def number(value):
        """把无定义指标显示为 N/A，其余按指定精度输出，避免把缺失写成零。"""
        return "N/A" if value is None else f"{value:.4f}"

    for c in results:
        low = "不复核" if c["review_lower"] is None else str(c["review_lower"])
        prefix = f"|{low}|{c['decision_threshold']}|{c['requested_reviews']}|"
        if c["status"] != "ok":
            lines.append(prefix + f"—|—|—|—|—|缺少{len(c['missing_reviews'])}条复核，无法评估|")
        else:
            m = c["metrics"]
            conf = m["confusion_matrix"]
            lines.append(
                prefix + f"{m['valid_predictions']}/{planned}|{conf['fn']}|{conf['fp']}|"
                f"{number(m['log_loss_on_valid'])}|{number(m['brier_score_on_valid'])}|可评估|"
            )
    lines += [
        "",
        "以下只在所有可评估方案的共同有效样本上比较，不能用它代替上表的覆盖率。",
        "",
        "|复核下界|分类阈值|共同FN|共同FP|共同loss|共同Brier|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in results:
        if c["status"] != "ok":
            continue
        m = c["common_valid_metrics"]
        conf = m["confusion_matrix"]
        low = "不复核" if c["review_lower"] is None else str(c["review_lower"])
        lines.append(
            f"|{low}|{c['decision_threshold']}|{conf['fn']}|{conf['fp']}|"
            f"{number(m['log_loss_on_valid'])}|{number(m['brier_score_on_valid'])}|"
        )
    lines += [
        "",
        "完整来源分层、覆盖率、分类正反向变化、已缓存复核用量及缺失UUID见 results.json。",
        "用量和耗时表示被选中的历史调用，不是本次产生的费用或网络时间。",
        "这是同一次随机输出的事后重放，不能代表重新调用必定得到相同结果。",
        "不自动选择最佳阈值。请在验证集选参，冻结后再用独立测试集评价。",
        "",
    ]
    return "\n".join(lines)


def run(args):
    """校验后生成独立输出目录与完整源快照；不改源运行、不覆盖旧回放。"""
    config, samples, frozen, blobs = load_run(args.run)
    upper = (config.get("review") or {}).get("upper_threshold") or config["decision_threshold"]
    if args.review_upper is not None:
        upper = args.review_upper
    if not finite_probability(upper) or upper <= 0:
        raise ValueError("Review upper must be finite and in (0, 1]")
    lowers = list(dict.fromkeys(args.review_thresholds))
    thresholds = list(dict.fromkeys(args.decision_thresholds))
    if not lowers or not thresholds or not all(finite_probability(t) for t in thresholds):
        raise ValueError("Provide finite decision thresholds in [0, 1] and review thresholds")
    for lower in lowers:
        validate_review_threshold(lower, upper)
    output = (
        args.output
        or (load_paths()["threshold_sweeps"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    ).resolve()
    if output.is_relative_to(args.run.resolve()):
        raise ValueError("Output must be outside the source run directory")
    results, common = sweep(frozen, samples, lowers, thresholds, upper)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name, data in blobs.items():
        (source / name).write_bytes(data)
    dump(output / "frozen_predictions.json", frozen)
    dump(output / "common_valid_images.json", common)
    manifest = dict(
        schema="he-frozen-threshold-sweep-v1",
        source_run=str(args.run.resolve()),
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
        review_thresholds=lowers,
        review_upper=upper,
        decision_thresholds=thresholds,
        planned=len(samples),
        common_valid=len(common),
        network_calls=0,
        policy="use_cached_review_when_lower_lt_p1_lt_upper_else_initial",
        missing_review_policy="skip_entire_candidate_no_fallback",
        initial_usage=token_usage([r["initial"] for r in frozen]),
    )
    dump(output / "manifest.json", manifest)
    for c in results:
        if c["status"] == "ok":
            directory = output / c["id"]
            directory.mkdir()
            rows = c.pop("predictions")
            (directory / "predictions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            c["predictions_file"] = f"{c['id']}/predictions.jsonl"
    dump(output / "results.json", results)
    text = summary_markdown(results, len(samples), len(common))
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to: {output}")
    return 0


def main():
    """解析源 run 与阈值网格，生成新目录；参数或快照校验失败返回退出码 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed source run directory")
    parser.add_argument(
        "--review-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help="Candidate review lower bounds; single-pass control always included",
    )
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5],
        help="Independent final classification thresholds; does not change review gate",
    )
    parser.add_argument(
        "--review-upper",
        type=float,
        default=None,
        help="Default: frozen source review upper, usually 0.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
