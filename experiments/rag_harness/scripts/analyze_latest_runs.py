"""读取最近八组真实运行；固定其文件指纹，离线复算共同队列及阈值权衡。"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from he_agent import metrics


def analyze(root):
    """按创建时间读取最近八个真实 run，对齐共同样本后计算总体与来源指标。"""
    candidates = []
    for path in (root / "runs").glob("*/run.json"):
        config = json.loads(path.read_text())
        if not config.get("dry_run") and config.get("created_utc"):
            candidates.append((config["created_utc"], path.parent, config))
    selected = sorted(candidates)[-8:]
    if len(selected) != 8:
        raise ValueError("Need eight live runs")
    records = []
    for date, folder, config in selected:
        samples = json.loads((folder / "samples.json").read_text())
        rows = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text().splitlines()
            if line
        ]
        assert len({r["image"] for r in rows}) == len(rows)
        lookup = {s["image"]: s for s in samples}
        assert all(r["true_label"] == lookup[r["image"]]["label"] for r in rows)
        files = ["run.json", "samples.json", "predictions.jsonl", "prompt.txt", "split.json"]
        records.append(
            dict(
                id=folder.name,
                config=config,
                samples=samples,
                rows=rows,
                hashes={n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in files},
                latest_order=[s["image"] for s in samples],
            )
        )
    assert all(r["latest_order"] == records[0]["latest_order"] for r in records)
    assert len({r["config"]["split_sha256"] for r in records}) == 1
    assert all(r["config"]["model_config"] == records[0]["config"]["model_config"] for r in records)
    common = set.intersection(
        *[{x["image"] for x in r["rows"] if x["status"] == "ok"} for r in records]
    )
    out = dict(
        protocol="latest-eight-live-runs-analysis-v1",
        selection="created_utc; non-dry-run",
        common_valid=len(common),
        same_order=True,
        records=[],
        knowledge_pairs=[],
    )
    for r in records:
        rows = [x for x in r["rows"] if x["image"] in common]
        c = r["config"]
        full = metrics(r["rows"], len(r["samples"]))
        saved = json.loads((root / "runs" / r["id"] / "metrics.json").read_text())
        for field in ("brier_score_on_valid", "log_loss_on_valid", "roc_auc_on_valid"):
            assert abs(full[field] - saved[field]) < 1e-12

        def source_metrics(items):
            """在传入的同一病例集合内分来源计算指标，单类别 AUC 仍保持无定义。"""
            return {
                s: metrics(
                    [x for x in items if x["source_dataset"] == s],
                    sum(x["source_dataset"] == s for x in items),
                )
                for s in sorted({x["source_dataset"] for x in items})
            }

        pos = [x["p_he"] for x in rows if x["true_label"] == 1]
        neg = [x["p_he"] for x in rows if x["true_label"] == 0]
        rec = dict(
            id=r["id"],
            mode=c["mode"],
            reference_count=c.get("reference_count", 0),
            created_utc=c["created_utc"],
            provider=c["provider"],
            model=c["model"],
            split_sha256=c["split_sha256"],
            prompt_sha256=c["prompt_sha256"],
            hashes=r["hashes"],
            full=full,
            common=metrics(rows, len(rows)),
            by_source_common=source_metrics(rows),
            class_probability=dict(
                positive_mean=statistics.mean(pos),
                negative_mean=statistics.mean(neg),
                positive_median=statistics.median(pos),
                negative_median=statistics.median(neg),
                positive_below_01=sum(p < 0.1 for p in pos),
                positive_01_to_03=sum(0.1 <= p < 0.3 for p in pos),
                positive_03_to_05=sum(0.3 <= p < 0.5 for p in pos),
            ),
            exploratory_thresholds={
                str(t): metrics(rows, len(rows), t) for t in (0.5, 0.4, 0.3, 0.2, 0.1)
            },
        )
        out["records"].append(rec)
    for base, knowledge, count in [
        ("baseline", "knowledge", 0),
        *[("reference", "reference_knowledge", n) for n in (4, 6, 8)],
    ]:
        a = next(
            r
            for r in records
            if r["config"]["mode"] == base and r["config"].get("reference_count", 0) == count
        )
        b = next(
            r
            for r in records
            if r["config"]["mode"] == knowledge and r["config"].get("reference_count", 0) == count
        )
        aa = {r["image"]: r for r in a["rows"] if r["status"] == "ok"}
        bb = {r["image"]: r for r in b["rows"] if r["status"] == "ok"}
        names = sorted(aa.keys() & bb.keys())
        ar = [aa[n] for n in names]
        br = [bb[n] for n in names]
        ma = metrics(ar, len(ar))
        mb = metrics(br, len(br))
        out["knowledge_pairs"].append(
            dict(
                base=base,
                count=count,
                n=len(names),
                delta_log_loss=mb["log_loss_on_valid"] - ma["log_loss_on_valid"],
                delta_brier=mb["brier_score_on_valid"] - ma["brier_score_on_valid"],
                delta_positive_mean=statistics.mean(
                    bb[n]["p_he"] - aa[n]["p_he"] for n in names if aa[n]["true_label"] == 1
                ),
                delta_negative_mean=statistics.mean(
                    bb[n]["p_he"] - aa[n]["p_he"] for n in names if aa[n]["true_label"] == 0
                ),
                extra_fn=mb["confusion_matrix"]["fn"] - ma["confusion_matrix"]["fn"],
                delta_fp=mb["confusion_matrix"]["fp"] - ma["confusion_matrix"]["fp"],
            )
        )
    return out


def main():
    """解析新输出路径并保存八组分析 JSON；不覆盖已有结果或调用模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(PROJECT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print("Common valid", result["common_valid"])
    for row in result["records"]:
        m = row["common"]
        print(
            row["id"],
            row["mode"],
            row["reference_count"],
            "valid",
            row["full"]["valid_predictions"],
            "loss",
            round(m["log_loss_on_valid"], 4),
            "brier",
            round(m["brier_score_on_valid"], 4),
            m["confusion_matrix"],
            row["class_probability"],
        )
    print("Paired knowledge changes:", result["knowledge_pairs"])


if __name__ == "__main__":
    main()
