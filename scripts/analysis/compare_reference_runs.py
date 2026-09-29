"""在预先冻结的共同测试集上比较参考选择，不将移入参考池的病例计为提升。

read_run 核验完整单次记录；compare 在冻结交集上算配对差值；main 写新结果。
只读历史预测，不读密钥、不调用 API、不改参考图或测试标签。
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
# 分析脚本复用主 Agent 的指标定义，避免两份公式随维护逐渐分叉。
sys.path.insert(0, str(PROJECT / "src"))
from he_agent import dump, metrics  # noqa: E402


def read_run(path):
    """读取完整运行并检查配置快照、重复项及概率协议，不接受 dry-run。"""
    config = json.loads((path / "run.json").read_text())
    if config.get("review", {}).get("enabled"):
        raise ValueError(
            "Reference-only comparison does not accept selective-review runs; "
            "use the paired initial/final metrics in metrics.json"
        )
    samples = json.loads((path / "samples.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    lookup = {r["image"]: r for r in samples}
    predictions = {r["image"]: r for r in rows}
    if config.get("dry_run"):
        raise ValueError("Cannot compare dry-runs")
    if (
        len(lookup) != len(samples)
        or len(predictions) != len(rows)
        or set(predictions) != set(lookup)
        or config["selected"] != len(samples)
    ):
        raise ValueError("Run is incomplete or contains duplicate/unplanned samples")
    split_bytes = (path / "split.json").read_bytes()
    if hashlib.sha256(split_bytes).hexdigest() != config["split_sha256"]:
        raise ValueError("Split snapshot hash mismatch")
    if hashlib.sha256((path / "prompt.txt").read_bytes()).hexdigest() != config["prompt_sha256"]:
        raise ValueError("Prompt snapshot hash mismatch")
    split = json.loads(split_bytes)
    refs = json.loads((path / "references.json").read_text())
    names = [r["image"] for r in refs]
    if (
        len(names) != config["reference_count"]
        or len(set(names)) != len(names)
        or not set(names) <= set(split["reference_pool"])
        or not set(lookup) <= set(split["test"])
        or set(split["reference_pool"]) & set(lookup)
    ):
        raise ValueError("Invalid reference/test membership")
    for row in rows:
        meta = lookup[row["image"]]
        if row["true_label"] != meta["label"] or row["source_dataset"] != meta["source_dataset"]:
            raise ValueError("Prediction truth/source mismatch")
        p = row.get("p_he")
        if row["status"] == "ok":
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Invalid probability")
        elif row["status"] not in ("abstain", "error"):
            raise ValueError("Invalid prediction status")
    return config, lookup, predictions, split


def compare(old_path, new_path, cohort_path):
    """按 UUID 对齐冻结病例，分别报告覆盖率与共同有效样本上的配对差值。"""
    cohort = json.loads(cohort_path.read_text())
    names = cohort["images"]
    if (
        cohort.get("schema") != "reference-comparison-cohort-v1"
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid comparison cohort")
    old_config, old_samples, old, old_split = read_run(old_path)
    new_config, new_samples, new, new_split = read_run(new_path)
    for key in (
        "model_config",
        "mode",
        "reference_count",
        "prompt_sha256",
        "labels_sha256",
        "seed",
        "decision_threshold",
    ):
        if old_config[key] != new_config[key]:
            raise ValueError(f"Uncontrolled comparison: different {key}")
    if old_config["split_sha256"] != cohort["parent_split_sha256"]:
        raise ValueError("Old run does not use the cohort's original split")
    if set(names) != set(old_split["test"]) & set(new_split["test"]):
        raise ValueError("Cohort is not the complete old/new test intersection")
    for image in names:
        if image not in old or image not in new:
            raise ValueError("Run does not cover the entire frozen comparison cohort")
        expected = cohort["samples"][image]
        for samples in (old_samples, new_samples):
            if any(samples[image][key] != expected[key] for key in ("label", "sha256")):
                raise ValueError("Cohort image/label fingerprint mismatch")
    threshold = old_config["decision_threshold"]
    common = [x for x in names if old[x]["status"] == new[x]["status"] == "ok"]
    if not common:
        raise ValueError("No common valid predictions")
    old_metrics = metrics([old[x] for x in common], len(common), threshold)
    new_metrics = metrics([new[x] for x in common], len(common), threshold)
    sources = sorted({old[x]["source_dataset"] for x in common})
    return {
        "old_run": str(old_path.resolve()),
        "new_run": str(new_path.resolve()),
        "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "planned_common": len(names),
        "coverage_on_frozen_cohort": {
            "old": metrics([old[x] for x in names], len(names), threshold),
            "new": metrics([new[x] for x in names], len(names), threshold),
        },
        "common_valid": len(common),
        "excluded_from_common_valid": [x for x in names if x not in set(common)],
        "common_valid_metrics": {"old": old_metrics, "new": new_metrics},
        "delta_log_loss_new_minus_old": new_metrics["log_loss_on_valid"]
        - old_metrics["log_loss_on_valid"],
        "delta_brier_new_minus_old": new_metrics["brier_score_on_valid"]
        - old_metrics["brier_score_on_valid"],
        "by_source_on_common_valid": {
            source: {
                name: metrics(
                    [rows[x] for x in common if rows[x]["source_dataset"] == source],
                    sum(old[x]["source_dataset"] == source for x in common),
                    threshold,
                )
                for name, rows in (("old", old), ("new", new))
            }
            for source in sources
        },
        "limitations": [
            "Exploratory paired comparison; references were selected after prior test-set analysis.",
            "Single runs cannot isolate model randomness; repeat and validate on independent data.",
            "No threshold or calibration was fitted on these test results.",
        ],
    }


def main():
    """输出复算 JSON；可指定新文件保存，拒绝覆盖已有结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite comparison output")
    result = compare(args.old_run, args.new_run, args.cohort)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
