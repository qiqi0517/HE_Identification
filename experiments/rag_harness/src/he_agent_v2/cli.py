"""固定参考图、离线检查与正式运行入口；所有新结果写入独立实验目录。"""

import argparse
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from he_agent import dump, load_samples
from models import create_model
from reference_split import validate_split

from . import LAB, PROJECT
from .evaluate import evaluate
from .fixed_references import FixedReferences, canonical, sha
from .harness import MODES, infer_one
from .knowledge_retrieval import KnowledgeLibrary
from .schemas import Query


def load_context(args):
    """校验全部标签与划分，但返回的推理对象随后会剥离真值。"""
    samples = load_samples(args.dataset)
    split = json.loads(args.split.read_text())
    tests, _ = validate_split(split, samples)
    return samples, split, tests


def run(args, client=None):
    """可注入模型进行无网络集成测试；正式入口复用现有适配器。"""
    if (
        args.limit < 0
        or not 0 <= args.retries <= 2
        or args.max_tokens <= 0
        or args.timeout <= 0
        or not math.isfinite(args.threshold)
        or not 0 <= args.threshold <= 1
    ):
        raise ValueError("Invalid numeric arguments")
    samples, split, tests = load_context(args)
    reference_set = FixedReferences(args.dataset, samples, split, args.reference_count)
    library = KnowledgeLibrary(LAB / "knowledge") if args.mode == "knowledge_review" else None
    prompts = {
        "basic": (PROJECT / "prompts/he_binary.txt").read_text(),
        **{name: (LAB / f"prompts/{name}.txt").read_text() for name in ("observe", "refine")},
    }
    protocol = json.loads((LAB / "config.json").read_text())
    expected = {
        "reference_strategy": "fixed",
        "default_reference_count": 8,
        "max_branches": 2,
        "max_cards": 2,
        "knowledge_character_budget": 600,
        "max_logical_calls": 2,
        "max_retries_per_stage": 2,
        "dhash_exclusion_distance": 4,
        "final_prediction": "stage2_only",
        "stage1_probability_forwarded": False,
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Protocol manifest differs from implemented frozen limits")
    random.Random(args.seed).shuffle(tests)
    if args.limit:
        tests = tests[: args.limit]
    client = client or create_model(
        args.provider, args.model, args.max_tokens, args.timeout, args.retries
    )
    if not args.dry_run:
        client.validate_credentials()
    output = args.output or LAB / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_{args.provider}_{args.mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshots"
    snapshot.mkdir()
    for name, text in prompts.items():
        (snapshot / f"{name}.txt").write_text(text)
    if library is not None:
        shutil.copytree(LAB / "knowledge", snapshot / "knowledge")
    dump(snapshot / "split.json", split)
    dump(snapshot / "references.json", reference_set.rows)
    dump(snapshot / "config.json", protocol)
    dump(output / "samples.json", tests)
    config = {
        "protocol": protocol,
        "mode": args.mode,
        "model": client.describe(),
        "seed": args.seed,
        "limit": args.limit,
        "threshold": args.threshold,
        "dry_run": args.dry_run,
        "dataset": str(args.dataset.resolve()),
        "split_sha256": sha(canonical(split)),
        "reference_strategy": "fixed",
        "reference_count": len(reference_set.rows),
        "references_sha256": reference_set.digest,
        "sample_order_sha256": sha(canonical([s["image"] for s in tests])),
        "knowledge_manifest": library.manifest if library is not None else None,
        "prompt_sha256": {k: sha(v.encode()) for k, v in prompts.items()},
        "code_sha256": {
            str(p.relative_to(LAB)): sha(p.read_bytes())
            for p in sorted((LAB / "src").rglob("*.py"))
        },
        "shared_code_sha256": {
            str(p.relative_to(PROJECT)): sha(p.read_bytes())
            for p in sorted((PROJECT / "src").rglob("*.py"))
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    dump(output / "run.json", config)
    for name in ("references", "stage1", "stage2", "predictions", "trace"):
        (output / f"{name}.jsonl").touch()
    results = []

    def append(name, row):
        """以 JSONL 追加事件；拒绝 NaN，保留逐阶段可回放记录。"""
        with (output / f"{name}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    try:
        for i, sample in enumerate(tests, 1):
            query = Query.from_sample(sample, args.dataset, split)

            def emit(kind, value):
                """附上当前图片身份并写总追踪；参考和两阶段事件另存便于分析。"""
                event = {"image": query.image, "event": kind, **value}
                append("trace", event)
                if kind in {"references", "stage1", "stage2"}:
                    append(kind, event)

            result = infer_one(
                query,
                reference_set,
                library,
                client,
                prompts,
                args.mode,
                args.retries,
                args.dry_run,
                emit,
            )
            # 推理已结束，才在本地合并真值和来源用于评价。
            row = {
                **result,
                "image": sample["image"],
                "true_label": sample["label"],
                "source_dataset": sample["source_dataset"],
            }
            row["p_non_he"] = 1 - row["p_he"] if row["status"] == "ok" else None
            row["predicted_label"] = (
                int(row["p_he"] >= args.threshold) if row["status"] == "ok" else None
            )
            results.append(row)
            append("predictions", row)
            dump(output / "metrics.json", evaluate(results, tests, args.threshold))
            print(
                f"[{i}/{len(tests)}] {sample['image']} truth={sample['label']} "
                f"p_he={row['p_he']} status={row['status']} "
                f"calls={len(row['calls'])}",
                flush=True,
            )
            if row["fatal"]:
                config["status"] = "fatal_stopped"
                break
        else:
            config["status"] = "dry_run_complete" if args.dry_run else "complete"
    finally:
        if config["status"] == "running":
            config["status"] = "interrupted"
        config["finished_at"] = datetime.now(timezone.utc).isoformat()
        dump(output / "run.json", config)
        summary = evaluate(results, tests, args.threshold)
        dump(output / "metrics.json", summary)
    print(
        f"Brier={summary['brier_score_on_valid']} loss={summary['log_loss_on_valid']} "
        f"coverage={summary['coverage']}\n结果：{output}"
    )
    return output


def parser():
    """构造历史原型的命令行参数，不启动实验或读取图片。"""
    root = argparse.ArgumentParser(
        description="Agent 2.0: fixed reference images + two-stage harness"
    )
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", type=Path, default=PROJECT / "data/unified")
    p.add_argument("--split", type=Path, default=PROJECT / "configs/reference_split.json")
    p.add_argument("--reference-count", type=int, choices=(4, 6, 8), default=8)
    p.add_argument("--mode", choices=MODES, default="review")
    p.add_argument("--provider", choices=("glm", "openai"), default="glm")
    p.add_argument("--model")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return root


def main():
    """解析命令并调用原型 run，将本地校验失败转换为清楚的退出信息。"""
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Local validation failed ({type(exc).__name__}): {exc}") from None
