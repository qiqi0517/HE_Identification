"""Seven-class auto-harness. Network operations require explicit --live."""

import argparse
import json
import logging
from pathlib import Path

from . import search
from . import workflow as w
from .artifacts import RunLock, read_json, seal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument(
        "--split", type=Path, default=Path("configs/auto-harness-typing/split_v1.json")
    )
    init.add_argument("--dataset", type=Path, default=Path("data/unified_typing_v2"))
    init.add_argument("--config", type=Path, default=Path("configs/auto_harness.example.json"))
    init.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    for name in (
        "collect",
        "search",
        "optimize",
        "shortlist",
        "select",
        "test",
        "status",
        "export",
        "budget",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--workspace", type=Path, required=True)
        if name == "collect":
            sub.add_argument("--role", choices=("search", "selection", "test"), required=True)
        if name in ("collect", "search"):
            sub.add_argument("--version", choices=("V0", "V1", "V2", "V3", "V4"), default="V0")
        if name in ("collect", "optimize"):
            sub.add_argument("--live", action="store_true")
        if name == "export":
            sub.add_argument("--output", type=Path, required=True)
        if name == "budget":
            for field in (
                "vision_calls",
                "vision_reported_tokens",
                "optimizer_calls",
                "optimizer_reported_tokens",
            ):
                sub.add_argument("--" + field.replace("_", "-"), type=int)
    pred = commands.add_parser("predict")
    pred.add_argument("--deployment", type=Path, required=True)
    pred.add_argument("--image", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--live", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.command == "init":
            exp = w.initialize(args.output, args.split, args.dataset, args.config)
            result = dict(
                workspace=str(args.output), experiment_id=exp["experiment_id"], network_calls=0
            )
        elif args.command == "demo":
            from .demo import run_demo

            result = run_demo(args.output)
        elif args.command == "predict":
            if args.output.exists():
                raise FileExistsError("Prediction output exists")
            result = w.predict(args.deployment, args.image, args.live)
            if args.live:
                seal(args.output, result)
        elif args.command == "status":
            exp = w.experiment(args.workspace)
            result = dict(
                experiment_id=exp["experiment_id"],
                simulation=exp["simulation"],
                budgets=read_json(args.workspace / "budget.json"),
                artifacts={
                    n: (args.workspace / n).exists()
                    for n in (
                        "search/V0/bank.json",
                        "search/V0/optimized.json",
                        "optimizer_history.jsonl",
                        "shortlist.json",
                        "selection/results.json",
                        "selected.json",
                        "test/results.json",
                        "test_report.json",
                    )
                },
            )
        else:
            with RunLock(args.workspace):
                if args.command == "collect":
                    result = w.collect(args.workspace, args.role, args.version, args.live)
                    if not result.get("dry_run"):
                        result = dict(saved=True, role=args.role, workspace=str(args.workspace))
                elif args.command == "search":
                    out = w.search(args.workspace, args.version)
                    result = {
                        k: {field: v["score"][field] for field in ("accuracy", "correct", "calls")}
                        for k, v in out["candidates"].items()
                    }
                elif args.command == "optimize":
                    result = search.run(args.workspace, args.live)
                elif args.command == "shortlist":
                    result = w.shortlist(args.workspace)
                elif args.command == "select":
                    out = w.select(args.workspace)
                    result = dict(
                        winner=out["winner"]["name"], accuracy=out["winner"]["score"]["accuracy"]
                    )
                elif args.command == "test":
                    out = w.test(args.workspace)
                    result = dict(
                        winner=out["winner"],
                        report=str(args.workspace / "test_report.json"),
                        accuracy={k: v["accuracy"] for k, v in out["scores"].items()},
                    )
                elif args.command == "export":
                    w.export(args.workspace, args.output)
                    result = dict(deployment=str(args.output), reference_images=7)
                elif args.command == "budget":
                    result = w.update_budget(
                        args.workspace,
                        {
                            k: getattr(args, k)
                            for k in (
                                "vision_calls",
                                "vision_reported_tokens",
                                "optimizer_calls",
                                "optimizer_reported_tokens",
                            )
                            if getattr(args, k) is not None
                        },
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        p.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
