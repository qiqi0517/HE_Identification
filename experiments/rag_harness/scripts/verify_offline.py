"""固定原图版两服务商三模式离线审计；阻止导入编码器依赖。"""

import argparse
import contextlib
import importlib.abc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments/rag_harness/src"))
from he_agent import dump, load_samples
from he_agent_v2 import LAB
from he_agent_v2.cli import parser, run
from he_agent_v2.fixed_references import FixedReferences
from he_agent_v2.schemas import Query
from reference_split import select_references, validate_split


class NoEncoderImports(importlib.abc.MetaPathFinder):
    """装有深度学习依赖的环境也要证明正常流程不会触发导入。"""

    def find_spec(self, fullname, path=None, target=None):
        """拦截编码器相关导入，确保固定原图原型不意外依赖 torch 等重组件。"""
        if fullname.split(".")[0] in {
            "torch",
            "torchvision",
            "timm",
            "numpy",
            "ultrasound_encoder",
        }:
            raise AssertionError("Encoder dependency imported: " + fullname)
        return None


def main():
    """在无模型调用的条件下验证原型模式与参考数矩阵，并输出本地检查记录。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    guard = NoEncoderImports()
    sys.meta_path.insert(0, guard)
    output = args.output or LAB / "runs" / (
        "fixed_verification_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "he-fixed-harness-v2",
        "live_api_calls": 0,
        "dry_runs": [],
        "reference_checks": [],
    }
    try:
        for provider in ("glm", "openai"):
            for mode in ("reference", "review", "knowledge_review"):
                directory = output / f"{provider}_{mode}"
                options = parser().parse_args(
                    [
                        "run",
                        "--provider",
                        provider,
                        "--mode",
                        mode,
                        "--reference-count",
                        "8",
                        "--dry-run",
                        "--output",
                        str(directory),
                    ]
                )
                with (output / "dryrun.log").open("a") as log, contextlib.redirect_stdout(log):
                    run(options)
                metrics = json.loads((directory / "metrics.json").read_text())
                config = json.loads((directory / "run.json").read_text())
                row = {
                    "provider": provider,
                    "mode": mode,
                    "samples": metrics["dry_run_samples"],
                    "errors": metrics["errors"],
                    "http_attempts": metrics["cost"]["http_attempts"],
                    "order": config["sample_order_sha256"],
                    "references": config["references_sha256"],
                }
                summary["dry_runs"].append(row)
                print(row, flush=True)
        samples = load_samples(PROJECT / "data/unified")
        for name in ("reference_split", "reference_split_diverse"):
            split = json.loads((PROJECT / f"configs/{name}.json").read_text())
            tests, _ = validate_split(split, samples)
            for count in (4, 6, 8):
                fixed = FixedReferences(PROJECT / "data/unified", samples, split, count)
                expected = select_references(split, samples, count)
                assert [r["image"] for r in fixed.rows] == [r["image"] for r in expected]
                for sample in tests:
                    refs, _ = fixed.prepare(
                        Query.from_sample(sample, PROJECT / "data/unified", split)
                    )
                    assert len(refs) == count
                summary["reference_checks"].append(
                    {"split": name, "count": count, "queries": len(tests), "errors": 0}
                )
        assert len({r["order"] for r in summary["dry_runs"]}) == 1
        assert len({r["references"] for r in summary["dry_runs"]}) == 1
        planned = len(json.loads((PROJECT / "configs/reference_split.json").read_text())["test"])
        assert all(
            r["samples"] == planned and r["errors"] == 0 and r["http_attempts"] == 0
            for r in summary["dry_runs"]
        )
        forbidden = {
            "torch",
            "torchvision",
            "timm",
            "numpy",
            "ultrasound_encoder",
        } & sys.modules.keys()
        assert not forbidden, forbidden
        summary["encoder_dependencies_imported"] = []
        dump(output / "summary.json", summary)
        print("All fixed-reference checks passed:", output)
    finally:
        sys.meta_path.remove(guard)


if __name__ == "__main__":
    main()
