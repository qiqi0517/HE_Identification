"""基础分型 Agent：四种输入模式，可选 confidence 中性复核，不执行策略优化。"""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from models import create_model
from project_paths import load_paths
from prompt_profiles import (
    KNOWLEDGE_MODES,
    MODES,
    REFERENCE_MODES,
    load_typing_prompt,
    load_typing_review_prompt,
)
from reference_split import validate_typing_split
from selective_review import (
    needs_confidence_review,
    predict_typing_stage,
    token_usage,
    validate_confidence_threshold,
)
from typing_protocol import CLASSES, SCHEMA, build_request, parse_prediction, report, unique_fields

PROJECT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_samples(dataset):
    dataset = Path(dataset).resolve()
    metadata = {}
    for line in (dataset / "labels/manifest.jsonl").read_text().splitlines():
        row = json.loads(line, object_pairs_hook=unique_fields)
        if row["image"] in metadata:
            raise ValueError("Duplicate manifest image")
        metadata[row["image"]] = row
    samples, seen = [], set()
    for line in (dataset / "labels/labels_typing.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, label = line.rsplit(maxsplit=1)
        if label not in CLASSES or name in seen:
            raise ValueError("Invalid or duplicate typing label")
        seen.add(name)
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset / "images") or not path.is_file():
            raise ValueError("Image missing or outside dataset/images")
        row = metadata[name]
        if row["typing_label"] != label or not row.get("primary_label_eligible", False):
            raise ValueError("Typing label disagrees with manifest")
        if digest(path.read_bytes()) != row["sha256"]:
            raise ValueError("Image hash mismatch")
        with Image.open(path) as im:
            im.load()
            if im.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported image format")
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[im.format]
            width, height = im.size
        sources = row.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(s, str) for s in sources)
        ):
            raise ValueError("Missing source metadata")
        samples.append(
            dict(
                image=name,
                label=label,
                sha256=row["sha256"],
                mime=mime,
                width=width,
                height=height,
                source_dataset="+".join(sorted(sources)),
            )
        )
    if not samples or seen != set(metadata):
        raise ValueError("Typing manifest and main labels must contain the same retained images")
    return sorted(samples, key=lambda s: s["image"])


def aggregate(results, selected):
    saved = report(results, selected)
    initial = [{**row, **row["initial_prediction"]} for row in results]
    saved["initial"] = report(initial, selected)
    reviewed = [r for r in results if r["reviewed"]]

    def correct(row):
        return row["status"] == "ok" and row["predicted_type"] == row["true_label"]

    saved["selective_review"] = {
        "reviewed": len(reviewed),
        "rescued_errors": sum(
            not correct({**r, **r["initial_prediction"]}) and correct(r) for r in reviewed
        ),
        "introduced_errors": sum(
            correct({**r, **r["initial_prediction"]}) and not correct(r) for r in reviewed
        ),
        "review_failures": sum(r["status"] != "ok" for r in reviewed),
        "usage_total": token_usage(
            [
                s
                for r in results
                for s in (r["initial_prediction"], r["review_prediction"])
                if s is not None
            ]
        ),
    }
    return saved


def run(args, model_client=None):
    if args.limit < 0 or args.max_tokens <= 0 or args.timeout <= 0 or args.retries < 0:
        raise ValueError("Invalid numeric argument")
    validate_confidence_threshold(args.review_threshold)
    if args.mode not in MODES:
        raise ValueError("Unknown mode")
    if args.reference_count is not None and args.mode not in REFERENCE_MODES:
        raise ValueError("--reference-count requires a reference mode")
    dataset = Path(args.dataset).resolve()
    all_samples = load_samples(dataset)
    split_path = Path(args.split).resolve()
    split = json.loads(split_path.read_text(), object_pairs_hook=unique_fields)
    samples, references = validate_typing_split(split, all_samples, args.role, args.reference_count)
    random.Random(args.seed).shuffle(samples)
    selected = samples[: args.limit] if args.limit else samples
    reference_inputs = []
    if args.mode in REFERENCE_MODES:
        for row in references:
            data = (dataset / row["image"]).read_bytes()
            if digest(data) != row["sha256"]:
                raise ValueError("Reference changed")
            reference_inputs.append(dict(bytes=data, mime=row["mime"], label=row["label"]))
    prompt, metadata, sources = load_typing_prompt(args.mode, args.prompt)
    review_reference = getattr(args, "review_reference", "same")
    review_knowledge = getattr(args, "review_knowledge", "checklist")
    if review_reference not in ("same", "none"):
        raise ValueError("Unknown review reference choice")
    review_inputs = reference_inputs if args.review and review_reference == "same" else []
    review_prompt, review_metadata, review_sources = None, {}, None
    if args.review:
        review_prompt, review_metadata, review_sources = load_typing_review_prompt(
            args.mode, review_knowledge, args.review_prompt, args.prompt
        )
    client = model_client or create_model(
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    model_info = client.describe()
    if not args.dry_run:
        client.validate_credentials()
    paths = load_paths()
    output = (
        Path(args.output)
        if args.output
        else paths["runs"] / ("typing_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "output_schema": SCHEMA,
        "model_config": model_info,
        "mode": args.mode,
        "dataset": str(dataset),
        "split_path": str(split_path),
        "role": args.role,
        "selected": len(selected),
        "eligible": len(samples),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "reference_count": len(reference_inputs),
        "uses_knowledge": args.mode in KNOWLEDGE_MODES,
        "labels_sha256": digest((dataset / "labels/labels_typing.txt").read_bytes()),
        "split_sha256": digest(split_path.read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review": {
            "enabled": args.review,
            "confidence_threshold": args.review_threshold,
            "condition": "status == ok and initial_confidence < threshold",
            "initial_prediction_sent_to_review": False,
            "final_policy": "replace_including_error_or_abstention",
            "max_logical_calls_per_image": 2 if args.review else 1,
            "prompt_sha256": digest(review_prompt.encode()) if review_prompt else None,
            "reference_policy": review_reference,
            "reference_count": len(review_inputs),
            "reference_images": [r["image"] for r in references] if review_inputs else [],
            **review_metadata,
        },
        **metadata,
    }
    dump(output / "run.json", config)
    dump(output / "split.json", split)
    dump(output / "samples.json", selected)
    dump(output / "references.json", references if reference_inputs else [])
    dump(output / "review_references.json", references if review_inputs else [])
    if review_sources:
        dump(output / "review_knowledge_sources.json", review_sources)
    (output / "prompt.txt").write_text(prompt)
    if review_prompt:
        (output / "review_prompt.txt").write_text(review_prompt)
    if sources:
        dump(output / "knowledge_sources.json", sources)
    if args.dry_run:
        dump(
            output / "validation.json",
            {
                "network_calls": 0,
                "checked_images": len(all_samples),
                "selected": len(selected),
                "class_counts": dict(Counter(s["label"] for s in selected)),
                "reference_images_per_request": len(reference_inputs),
                "review_reference_images_per_request": len(review_inputs),
                "review_knowledge": review_knowledge if args.review else None,
                "reference_class_counts": dict(Counter(r["label"] for r in references))
                if reference_inputs
                else {},
                "max_logical_calls_if_live": len(selected) * (2 if args.review else 1),
            },
        )
        print(f"Local typing check passed: {len(selected)} images; no API calls. {output}")
        return 0
    results = []
    with (
        (output / "predictions.jsonl").open("w") as log,
        (output / "stages.jsonl").open("w") as stages_log,
    ):
        for sample in selected:
            try:
                data = (dataset / sample["image"]).read_bytes()
                if digest(data) != sample["sha256"]:
                    raise ValueError("Image changed after validation")
                initial = predict_typing_stage(
                    client,
                    build_request(data, sample["mime"], prompt, reference_inputs),
                    parse_prediction,
                )
            except (OSError, ValueError):
                initial = dict(
                    status="error",
                    predicted_type=None,
                    confidence=None,
                    findings=[],
                    fatal=True,
                    error="Input image unavailable or changed",
                    usage=None,
                )
            stages_log.write(
                json.dumps(
                    {"image": sample["image"], "stage": "initial", **initial}, ensure_ascii=False
                )
                + "\n"
            )
            stages_log.flush()
            reviewed = None
            if args.review and needs_confidence_review(initial, args.review_threshold):
                print(
                    f"  review: initial_confidence={initial['confidence']:.4f} < {args.review_threshold}",
                    flush=True,
                )
                reviewed = predict_typing_stage(
                    client,
                    build_request(data, sample["mime"], review_prompt, review_inputs),
                    parse_prediction,
                )
                stages_log.write(
                    json.dumps(
                        {"image": sample["image"], "stage": "review", **reviewed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stages_log.flush()
            final = reviewed if reviewed is not None else initial
            row = dict(
                image=sample["image"],
                true_label=sample["label"],
                source_dataset=sample["source_dataset"],
                output_schema=SCHEMA,
                **final,
                initial_prediction=initial,
                review_prediction=reviewed,
                reviewed=reviewed is not None,
                final_stage="review" if reviewed else "initial",
                logical_calls=2 if reviewed else 1,
            )
            results.append(row)
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            saved = aggregate(results, selected)
            dump(output / "metrics.json", saved)
            print(
                f"{len(results)}/{len(selected)}: true_label={sample['label']}, predicted_type={final['predicted_type']}, confidence={final['confidence']}, status={final['status']}, reviewed={row['reviewed']}",
                flush=True,
            )
            if final.get("fatal"):
                break
    print(
        f"Final: accuracy={saved['accuracy']:.4f}, valid={saved['valid_predictions']}/{saved['planned']}, reviewed={saved['selective_review']['reviewed']}. Results: {output.resolve()}",
        flush=True,
    )
    return int(any(r["status"] == "error" for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/unified_typing_v2")
    parser.add_argument(
        "--split", type=Path, default=PROJECT / "configs/auto-harness-typing/split_v1.json"
    )
    parser.add_argument(
        "--role",
        choices=("search", "selection", "test"),
        default="search",
        help="Default search avoids using the reserved test set during development",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=None,
        help="Default all frozen reference images; otherwise take this many in frozen order",
    )
    parser.add_argument("--prompt", type=Path)
    parser.add_argument(
        "--review-prompt",
        type=Path,
        help="Override neutral review instructions; selected knowledge is appended",
    )
    parser.add_argument(
        "--review-reference",
        choices=("none", "same"),
        default="same",
        help="Review references: none, or the same images/order used for initial prediction",
    )
    parser.add_argument(
        "--review-knowledge",
        choices=("none", "same", "checklist"),
        default="checklist",
        help="Review knowledge: none, inherit initial knowledge, or use the independent checklist (default)",
    )
    parser.add_argument("--review", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.8,
        help="Review valid initial predictions with confidence < this value; equality does not trigger review",
    )
    parser.add_argument("--provider", default="glm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=3, help="0 = all images in the selected role")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
