"""CLI for preparation, nine training runs, selection and frozen evaluation."""

import argparse
import json
from pathlib import Path

from .data import PROJECT, prepare


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Map the existing split; reads metadata only")
    prep.add_argument("--dataset", type=Path, default=PROJECT / "data/unified_typing_v2")
    prep.add_argument(
        "--source-split", type=Path, default=PROJECT / "configs/auto-harness-typing/split_v1.json"
    )
    prep.add_argument(
        "--output", type=Path, default=PROJECT / "configs/expert-typing/v1/split.json"
    )
    prep.add_argument("--overwrite", action="store_true")
    training = sub.add_parser("train", help="Train one configuration/seed")
    training.add_argument(
        "--split", type=Path, default=PROJECT / "configs/expert-typing/v1/split.json"
    )
    training.add_argument("--dataset", type=Path)
    training.add_argument("--variant", choices=("S1", "S2", "S3"), required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--tiny-weights", type=Path, default=PROJECT / "experiments/expert_lab/weights/TinyUSFM.pth"
    )
    training.add_argument(
        "--cache", type=Path, default=PROJECT / "runs/expert_typing/feature_cache"
    )
    training.add_argument("--output", type=Path, required=True)
    training.add_argument(
        "--smoke", action="store_true", help="Seven train-only images; never eligible for selection"
    )
    training.add_argument(
        "--epochs", type=positive, help="Custom cap; marks run ineligible for v1 selection"
    )
    training.add_argument("--feature-scaling", choices=("none", "standard"), default="none")
    training.add_argument("--weight-decay", type=float)
    training.add_argument("--head-lr", type=float)
    training.add_argument("--finetune-head-lr", type=float)
    training.add_argument("--encoder-lr", type=float)
    training.add_argument("--early-stop-metric", choices=("accuracy", "loss"), default="accuracy")
    training.add_argument("--patience", type=positive)
    training.add_argument("--class-weight", choices=("none", "inverse_sqrt"), default="none")
    choosing = sub.add_parser(
        "select", help="Require all three configurations x three seeds and freeze winner"
    )
    choosing.add_argument("--runs", type=Path, default=PROJECT / "runs/expert_typing/v1")
    choosing.add_argument(
        "--output", type=Path, default=PROJECT / "runs/expert_typing/v1/selection.json"
    )
    choosing.add_argument(
        "--candidates",
        type=Path,
        help="Predeclared adjusted candidates; each requires seeds 42,43,44",
    )
    evaluating = sub.add_parser(
        "evaluate", help="Evaluate a checkpoint authorized by the frozen selection"
    )
    evaluating.add_argument("--freeze", type=Path, required=True)
    evaluating.add_argument("--dataset", type=Path)
    evaluating.add_argument(
        "--variant", choices=("S1", "S2", "S3"), help="Optional predeclared comparison model"
    )
    evaluating.add_argument("--candidate", help="Named adjusted candidate in the frozen selection")
    evaluating.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    evaluating.add_argument("--output", type=Path, required=True)
    predicting = sub.add_parser("predict", help="Predict one new image without generating findings")
    predicting.add_argument("--checkpoint", type=Path, required=True)
    predicting.add_argument("--image", type=Path, required=True)
    comparison = sub.add_parser(
        "compare", help="Compare completed same-image test runs, expert first"
    )
    comparison.add_argument("--first", type=Path, required=True)
    comparison.add_argument("--second", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    for child in (training, evaluating, predicting):
        child.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        child.add_argument("--threads", type=positive, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(args.dataset, args.source_split, args.output, args.overwrite)
    elif args.command == "train":
        from .training import train

        result = train(args)
    else:
        from .workflow import compare, evaluate, predict, select

        if args.command == "select":
            result = select(args.runs, args.output, args.candidates)
        elif args.command == "evaluate":
            result = evaluate(args)
        elif args.command == "predict":
            result = predict(args)
        else:
            result = compare(args.first, args.second, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
