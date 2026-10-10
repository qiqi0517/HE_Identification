"""Validation selection, frozen test evaluation and single-image prediction."""

import json
import statistics
import time
from pathlib import Path

import torch

from . import CLASSES, SCHEMA
from .data import code_hash, image_path, load_split, read, rows_for, sha, write, write_lines
from .metrics import paired_comparison, score
from .network import device_for, sync, tensor
from .training import load_checkpoint, prediction


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select(runs, output, candidates=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    entries, comparison, common = [], [], None
    specs = (
        read(candidates)["candidates"]
        if candidates
        else [dict(name=v, variant=v, expected={}) for v in ("S1", "S2", "S3")]
    )
    names = [spec["name"] for spec in specs]
    if (
        not specs
        or len(set(names)) != len(names)
        or any(
            not name
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in name
            )
            for name in names
        )
    ):
        raise ValueError("Candidate names must be unique safe directory names")
    for spec in specs:
        variant, name = spec["variant"], spec["name"]
        metrics_list = []
        training_identity = None
        for seed in (42, 43, 44):
            path = Path(runs) / name / str(seed)
            config, status = read(path / "config.json"), read(path / "status.json")
            if (
                config["smoke"]
                or status["status"] != "complete"
                or (not candidates and not config["protocol_eligible"])
            ):
                raise ValueError(f"Incomplete or smoke/custom run: {path}")
            if (
                config["variant"] != variant
                or config["seed"] != seed
                or config["code_sha256"] != code_hash()
            ):
                raise ValueError("Run identity/code mismatch")
            if any(config.get(k) != v for k, v in spec.get("expected", {}).items()):
                raise ValueError("Run differs from the declared candidate parameters")
            parameters = {
                k: config.get(k)
                for k in (
                    "variant",
                    "epochs",
                    "weight_decay",
                    "learning_rates",
                    "warmup_epochs",
                    "patience",
                    "feature_scaling",
                    "class_weight",
                    "class_weights",
                    "early_stop_metric",
                    "batch_size",
                    "pretrained_sha256",
                    "loss",
                )
            }
            if training_identity is not None and parameters != training_identity:
                raise ValueError("Seeds of a candidate use different hyperparameters")
            training_identity = parameters
            identity = (
                config["split_sha256"],
                config["code_sha256"],
                config["dataset_hashes"],
                config["image_version"],
                config["preprocess"],
            )
            if common is not None and identity != common:
                raise ValueError("Runs do not share the frozen experiment protocol")
            common = identity
            split, _ = load_split(path / "split.json")
            if sha(path / "split.json") != config["split_sha256"]:
                raise ValueError("Split snapshot changed")
            checkpoint_path = path / "best.pt"
            if sha(checkpoint_path) != status["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash changed")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload["config"] != config or payload["classes"] != list(CLASSES):
                raise ValueError("Checkpoint/run metadata mismatch")
            rows = rows_for(split, "validation")
            predictions = jsonl(path / "validation_predictions.jsonl")
            metrics = score(rows, predictions)
            if not metrics["complete"] or metrics["valid_predictions"] != len(rows):
                raise ValueError("Incomplete validation predictions")
            if metrics != read(path / "validation_metrics.json"):
                raise ValueError("Validation metrics changed")
            entries.append(
                dict(
                    variant=variant,
                    candidate=name,
                    seed=seed,
                    checkpoint=str(checkpoint_path.resolve()),
                    checkpoint_sha256=sha(checkpoint_path),
                    run=str(path.resolve()),
                )
            )
            metrics_list.append(metrics)
        comparison.append(
            dict(
                variant=variant,
                candidate=name,
                correct_sum=sum(m["correct"] for m in metrics_list),
                mean_accuracy=statistics.mean(m["accuracy"] for m in metrics_list),
                std_accuracy=statistics.stdev(m["accuracy"] for m in metrics_list),
                mean_macro_recall=statistics.mean(m["macro_recall"] for m in metrics_list),
            )
        )
    winner = max(
        comparison,
        key=lambda x: (
            x["correct_sum"],
            x["mean_macro_recall"],
            -names.index(x["candidate"]),
        ),
    )
    chosen = next(e for e in entries if e["candidate"] == winner["candidate"] and e["seed"] == 42)
    freeze = dict(
        schema=SCHEMA,
        code_sha256=code_hash(),
        selection_rule="sum_correct,mean_macro_recall,declared_candidate_order;deployment_seed42",
        candidate_manifest=read(candidates) if candidates else None,
        candidate_manifest_sha256=sha(candidates) if candidates else None,
        selected=chosen,
        candidates=entries,
        comparison=comparison,
        split=split,
        split_sha256=common[0],
        audit_status=split["audit_status"],
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write(output, freeze)
    return {"selected": chosen, "comparison": comparison, "freeze_sha256": sha(output)}


def infer_one(model, path, device):
    sync(device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = prediction(model(tensor(path).unsqueeze(0).to(device))[0])
        sync(device)
        result["elapsed_seconds"] = time.perf_counter() - start
    except Exception as exc:
        result = dict(
            status="error",
            predicted_type=None,
            confidence=None,
            confidence_kind="softmax_uncalibrated",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - start,
        )
    return result


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    frozen = read(args.freeze)
    if frozen["schema"] != SCHEMA or frozen["code_sha256"] != code_hash():
        raise ValueError("Freeze schema/code changed")
    candidate_name = getattr(args, "candidate", None)
    if candidate_name is None and args.variant is None:
        selected = frozen["selected"]
    else:
        matches = [
            e
            for e in frozen["candidates"]
            if e["seed"] == args.seed
            and (
                e.get("candidate", e["variant"]) == candidate_name
                if candidate_name
                else e["variant"] == args.variant
            )
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous/unknown checkpoint; select its --candidate name")
        selected = matches[0]
    path = Path(selected["checkpoint"])
    if sha(path) != selected["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(path, device)
    split = frozen["split"]
    from .data import validate_mapping

    root = Path(args.dataset or split["dataset"])
    validate_mapping(split, root)
    if payload["config"]["split_sha256"] != frozen["split_sha256"]:
        raise ValueError("Checkpoint does not belong to frozen split")
    rows = rows_for(split, "test")
    output.mkdir(parents=True)
    write(output / "samples.json", rows)
    write(
        output / "run.json",
        dict(
            schema=SCHEMA,
            role="test",
            variant=selected["variant"],
            candidate=selected.get("candidate", selected["variant"]),
            seed=selected["seed"],
            freeze_sha256=sha(args.freeze),
            checkpoint_sha256=sha(path),
            code_sha256=code_hash(),
            image_version=split["image_version"],
            split_sha256=frozen["split_sha256"],
            device=str(device),
            batch_size=1,
            audit_status=split["audit_status"],
            timing="one synthetic warmup; per-image decode+preprocess+inference, synchronized; excludes model load and hash check",
        ),
    )
    write(output / "status.json", dict(status="running"))
    predictions = []
    write(output / "metrics.json", score(rows, predictions))
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 224, 224, device=device))
        sync(device)
        for row in rows:
            source = image_path(root, row["image"])
            # A changed input invalidates the experiment, rather than silently scoring a new image.
            if sha(source) != row["sha256"]:
                raise ValueError(f"Test image hash changed: {row['image']}")
            result = dict(
                image=row["image"], true_label=row["label"], **infer_one(model, source, device)
            )
            predictions.append(result)
            write_lines(output / "predictions.jsonl", predictions)
            write(output / "metrics.json", score(rows, predictions))
        elapsed = [p["elapsed_seconds"] for p in predictions]
        summary = dict(
            status="complete",
            planned=len(rows),
            attempted=len(predictions),
            mean_seconds=statistics.mean(elapsed),
            median_seconds=statistics.median(elapsed),
            confidence_kind="softmax_uncalibrated",
            checkpoint_sha256=sha(path),
        )
        if device.type == "cuda":
            summary["peak_accelerator_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        else:
            summary["peak_accelerator_allocated_bytes"] = None
        try:
            import resource
            import sys

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        except ImportError:
            summary["process_peak_rss_bytes"] = None
        write(output / "status.json", summary)
        return summary
    except BaseException as exc:
        write(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


def predict(args):
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(args.checkpoint, device)
    return dict(
        schema=SCHEMA,
        model_version=f"{payload['config']['variant']}-seed{payload['config']['seed']}",
        checkpoint_sha256=sha(args.checkpoint),
        classes=list(CLASSES),
        **infer_one(model, Path(args.image), device),
    )


def compare(first, second, output):
    """Compare a completed expert test run with another same-image run (including LLM)."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    a, b = Path(first), Path(second)
    rows = read(a / "samples.json")
    other = read(b / "samples.json")
    for path in (a, b):
        if read(path / "run.json").get("role") != "test":
            raise ValueError("Comparison is only for frozen test runs")

    def identity(rs):
        return {r["image"]: (r["label"], r["sha256"]) for r in rs}

    if identity(rows) != identity(other):
        raise ValueError("Image version, labels or membership differ")
    for row in rows:
        if not row.get("group_id"):
            raise ValueError("First run must include expert association groups")
    first_predictions, second_predictions = (
        jsonl(a / "predictions.jsonl"),
        jsonl(b / "predictions.jsonl"),
    )
    result = dict(
        first=str(a),
        second=str(b),
        first_metrics=score(rows, first_predictions),
        second_metrics=score(rows, second_predictions),
        paired=paired_comparison(rows, first_predictions, second_predictions),
    )
    write(output, result)
    return result["paired"]
