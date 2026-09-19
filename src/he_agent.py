"""肝包虫病概率评估入口：校验数据、组织提示、调用模型并记录实验结果。"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from models import (
    ImagePart,
    ModelClient,
    ModelError,
    ModelRequest,
    ModelResponse,
    TextPart,
    create_model,
)
from prompt_profiles import KNOWLEDGE_MODES, MODES, REFERENCE_MODES, load_prompt

PROJECT = Path(__file__).resolve().parents[1]


def dump(path, value):
    """以 UTF-8 保存可读的 JSON 快照，便于检查与复现实验。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_samples(dataset):
    """仅读取主标签表中的图片，并校验路径、标签、哈希和图像解码。

    返回按图片路径排序的元数据；不推断未标注图片的诊断标签。
    """
    dataset = dataset.resolve()
    metadata = {
        r["image"]: r
        for r in (
            json.loads(x) for x in (dataset / "labels/manifest.jsonl").read_text().splitlines()
        )
    }
    rows, seen = [], set()
    for line in (dataset / "labels/labels.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, value = line.rsplit(None, 1)
        if value not in {"0", "1"} or rel in seen:
            raise ValueError("Invalid or duplicate binary label entry")
        seen.add(rel)
        path = (dataset / rel).resolve()
        if not path.is_relative_to(dataset / "images") or not path.is_file():
            raise ValueError("Image missing or outside dataset/images")
        meta = metadata[rel]
        if meta["label"] != int(value):
            raise ValueError("Label/manifest mismatch")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError("Image SHA256 mismatch")
        with Image.open(path) as im:
            im.load()
            if im.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Unsupported image format; no automatic conversion")
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[im.format]
            width, height = im.size
        rows.append(
            dict(
                image=rel,
                label=int(value),
                sha256=digest,
                mime=mime,
                width=width,
                height=height,
                source_dataset=meta.get("source_dataset"),
                normal_source=meta.get("source_normal_label") == 0,
            )
        )
    if not rows:
        raise ValueError("No labeled images")
    return sorted(rows, key=lambda r: r["image"])


def build_request(image_bytes, mime, prompt, references=()):
    """构造模型无关请求：参考图可带标签，测试图只发送原始图像。"""
    content = []
    if references:
        content.append(
            TextPart(
                "以下是带标签的参考超声图，仅用于学习类别示例；最后一张是待判断图片。不要输出参考图的判断。"
            )
        )
        for i, r in enumerate(references, 1):
            content.append(TextPart(f"参考图 {i}，诊断标签：{r['label']}。"))
            content.append(ImagePart(data=r["bytes"], mime=r["mime"]))
        content.append(
            TextPart(
                "以下为唯一待判断的测试图，标签未知。仅估计这张图的患病概率，返回要求的 JSON："
            )
        )
    else:
        content.append(TextPart("请估计这张肝脏超声图像对应的患病概率，按要求返回 JSON。"))
    content.append(ImagePart(data=image_bytes, mime=mime))
    return ModelRequest(system_prompt=prompt, content=tuple(content))


def parse_prediction(response: ModelResponse):
    """校验最终回答的概率协议；拒绝截断、重复字段及非法数值。

    允许 p_he=null 表示无法估计，不将其转换为阴性或 0.5。
    """
    if not response.completed:
        raise ValueError("Response did not finish normally (possibly max_tokens exhausted)")
    text = response.text
    if not isinstance(text, str):
        raise ValueError("Missing text content")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()

    def unique_fields(pairs):
        """拒绝重复 JSON 键，避免后一个概率静默覆盖前一个。"""
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON field")
            obj[key] = value
        return obj

    obj = json.loads(text, object_pairs_hook=unique_fields)
    if not isinstance(obj, dict) or set(obj) != {"p_he", "evidence"}:
        raise ValueError("Expected exactly p_he and evidence")
    probability = obj["p_he"]
    if probability is not None and (
        type(probability) not in (int, float)
        or not 0 <= probability <= 1
        or not math.isfinite(probability)
    ):
        raise ValueError("p_he must be a finite number in [0, 1] or null")
    if not isinstance(obj["evidence"], str) or not obj["evidence"].strip():
        raise ValueError("Missing evidence")
    return obj


def metrics(results, planned, threshold=0.5):
    """在有效概率上评分，并单独统计拒答、失败和未完成样本。

    threshold 只影响辅助分类；Brier、对数损失和 AUC 使用原始概率。
    """
    valid = [r for r in results if r["status"] == "ok"]
    c = Counter((r["true_label"], int(r["p_he"] >= threshold)) for r in valid)
    probabilities = [r["p_he"] for r in valid]
    truth = [r["true_label"] for r in valid]
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, truth)) / len(valid) if valid else None
    # 只在取对数时截断概率；保存结果与 Brier/AUC 仍使用原始数值。
    epsilon = 1e-15
    clipped = [min(max(p, epsilon), 1 - epsilon) for p in probabilities]
    log_loss = (
        -sum(y * math.log(p) + (1 - y) * math.log1p(-p) for p, y in zip(clipped, truth))
        / len(valid)
        if valid
        else None
    )
    positives = [p for p, y in zip(probabilities, truth) if y == 1]
    negatives = [p for p, y in zip(probabilities, truth) if y == 0]
    # AUC 等价于阳性概率高于阴性的配对比例；并列计半分，单类别时无定义。
    auc = (
        (
            sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives)
            / (len(positives) * len(negatives))
        )
        if positives and negatives
        else None
    )
    tn, fp, fn, tp = c[0, 0], c[0, 1], c[1, 0], c[1, 1]

    def div(a, b):
        """分母为零时返回 None，避免把无定义指标记成零分。"""
        return a / b if b else None

    return {
        "output_schema": "he-probability-v2",
        "decision_threshold": threshold,
        "probability_status": "uncalibrated_model_estimate",
        "brier_score_on_valid": brier,
        "log_loss_on_valid": log_loss,
        "roc_auc_on_valid": auc,
        "log_loss_clip_epsilon": epsilon,
        "planned": planned,
        "attempted": len(results),
        "valid_predictions": len(valid),
        "abstentions": sum(r["status"] == "abstain" for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "unattempted": planned - len(results),
        "coverage": div(len(valid), planned),
        "accuracy_on_valid": div(tp + tn, len(valid)),
        "correct_over_planned": div(tp + tn, planned),
        "precision_on_valid": div(tp, tp + fp),
        "sensitivity_on_valid": div(tp, tp + fn),
        "specificity_on_valid": div(tn, tn + fp),
        "f1_on_valid": div(2 * tp, 2 * tp + fp + fn),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def report(results, samples, threshold=0.5):
    """汇总总体与各数据来源的指标，便于识别来源差异。"""
    overall = metrics(results, len(samples), threshold)
    groups = sorted({r["source_dataset"] for r in samples})
    overall["by_source"] = {
        s: metrics(
            [r for r in results if r["source_dataset"] == s],
            sum(r["source_dataset"] == s for r in samples),
            threshold,
        )
        for s in groups
    }
    return overall


def run(args, model_client: ModelClient | None = None):
    """执行一次独立实验；可注入模拟客户端进行离线验证。

    先校验数据与配置，再创建新输出目录；逐图保存结果以保留中断前记录。
    """
    if args.limit < 0 or args.max_tokens <= 0 or args.timeout <= 0 or args.retries < 0:
        raise ValueError("Invalid numeric argument")
    threshold = getattr(args, "threshold", 0.5)
    if (
        type(threshold) not in (float, int)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Threshold must be a finite number in [0, 1]")
    samples = load_samples(args.dataset)
    all_samples = samples
    split_path = getattr(args, "split", None)
    mode = getattr(args, "mode", "baseline")
    if mode not in MODES:
        raise ValueError(f"Unknown experiment mode: {mode}")
    reference_count = getattr(args, "reference_count", None)
    if reference_count is not None and mode not in REFERENCE_MODES:
        raise ValueError("--reference-count requires reference or reference_knowledge mode")
    references = []
    split = None
    if mode in REFERENCE_MODES | KNOWLEDGE_MODES and split_path is None:
        raise ValueError(
            "Reference, knowledge and combined modes require --split for the fixed test set"
        )
    if split_path is not None:
        from reference_split import validate_split

        split = json.loads(split_path.read_text())
        samples, references = validate_split(split, samples)
    reference_inputs = []
    if mode in REFERENCE_MODES:
        from reference_split import select_references

        references = select_references(split, all_samples, reference_count)
        for r in references:
            data = (args.dataset / r["image"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != r["sha256"]:
                raise ValueError("Reference image changed")
            reference_inputs.append({"bytes": data, "mime": r["mime"], "label": r["label"]})
    # 各模式使用相同种子和固定测试集，使选取的样本及顺序一致。
    random.Random(args.seed).shuffle(samples)
    selected = samples[: args.limit] if args.limit else samples
    prompt, prompt_metadata, knowledge_sources = load_prompt(mode, getattr(args, "prompt", None))
    client = (
        model_client
        if model_client is not None
        else create_model(
            provider=getattr(args, "provider", "glm"),
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    )
    model_info = client.describe()
    if not args.dry_run:
        client.validate_credentials()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "output_schema": "he-probability-v2",
        "probability_status": "uncalibrated_model_estimate",
        "decision_threshold": threshold,
        "provider": model_info["provider"],
        "model": model_info["model"],
        "model_config": model_info,
        "dry_run": args.dry_run,
        "dataset": str(args.dataset.resolve()),
        "available_labeled": len(all_samples),
        "mode": mode,
        "uses_references": mode in REFERENCE_MODES,
        "reference_count": len(reference_inputs),
        "uses_knowledge": mode in KNOWLEDGE_MODES,
        "eligible_test": len(samples),
        "selected": len(selected),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
        "retries": args.retries,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(
            (args.dataset / "labels/labels.txt").read_bytes()
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    config.update(prompt_metadata)
    if knowledge_sources is not None:
        dump(args.output / "knowledge_sources.json", knowledge_sources)
    if split is not None:
        config["split_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        dump(args.output / "split.json", split)
        dump(
            args.output / "references.json",
            references if mode in REFERENCE_MODES else [],
        )
    dump(args.output / "run.json", config)
    (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")
    dump(args.output / "samples.json", selected)
    # dry-run 到此结束：仅做本地校验，不调用模型或伪造预测。
    if args.dry_run:
        dump(
            args.output / "validation.json",
            {
                "checked_labeled_images": len(all_samples),
                "selected": len(selected),
                "class_counts_all": dict(Counter(r["label"] for r in samples)),
                "network_calls": 0,
                "mode": mode,
                "reference_images_per_request": len(reference_inputs),
            },
        )
        print(f"Local check passed: {len(samples)} images; no API calls. {args.output}")
        return 0
    results = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as log:
        for sample in selected:
            start = time.monotonic()
            r = {
                "image": sample["image"],
                "true_label": sample["label"],
                "source_dataset": sample["source_dataset"],
                "normal_source": sample["normal_source"],
                "p_he": None,
                "p_non_he": None,
                "predicted_label": None,
                "decision_threshold": threshold,
                "output_schema": "he-probability-v2",
                "probability_status": "uncalibrated_model_estimate",
                "status": "error",
            }
            fatal = False
            try:
                data = (args.dataset / sample["image"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != sample["sha256"]:
                    raise ValueError("Image changed after validation")
                request = build_request(data, sample["mime"], prompt, reference_inputs)
                response = client.generate(request)
                r["response_id"] = response.response_id
                r["returned_model"] = response.model
                r["usage"] = response.usage
                r["finish_reason"] = response.finish_reason
                r["raw_answer"] = response.text
                prediction = parse_prediction(response)
                probability = prediction["p_he"]
                r.update(
                    p_he=probability,
                    p_non_he=None if probability is None else 1 - probability,
                    predicted_label=None if probability is None else int(probability >= threshold),
                    evidence=prediction["evidence"],
                    status="abstain" if probability is None else "ok",
                )
            except ModelError as e:
                r["error"] = str(e)
                fatal = e.fatal
            except (ValueError, KeyError, IndexError, TypeError) as e:
                r["error"] = f"Invalid input/response ({type(e).__name__})"
            r["latency_seconds"] = round(time.monotonic() - start, 3)
            results.append(r)
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()
            dump(args.output / "metrics.json", report(results, selected, threshold))
            summary = (
                f"p_he={r['p_he']:.4f}, p_non_he={r['p_non_he']:.4f}, "
                f"predicted_label={r['predicted_label']}, "
                f"correct={r['predicted_label'] == r['true_label']}"
                if r["status"] == "ok"
                else f"status={r['status']}, predicted_label=null"
            )
            # 真值仅用于本地展示和评分，不加入发送给模型的请求。
            print(
                f"{len(results)}/{len(selected)}: true_label={r['true_label']}, {summary}",
                flush=True,
            )
            if fatal:
                break
    # 与 metrics.json 使用同一评分函数；拒答和失败不伪造为零损失。
    final_metrics = metrics(results, len(selected), threshold)
    loss = final_metrics["log_loss_on_valid"]
    brier = final_metrics["brier_score_on_valid"]
    loss_text = "N/A" if loss is None else f"{loss:.6f}"
    brier_text = "N/A" if brier is None else f"{brier:.6f}"
    print(
        f"Final: loss(log_loss)={loss_text}, brier={brier_text}, "
        f"valid={final_metrics['valid_predictions']}/{final_metrics['planned']}, "
        f"abstain={final_metrics['abstentions']}, errors={final_metrics['errors']}, "
        f"unattempted={final_metrics['unattempted']}",
        flush=True,
    )
    print(f"Results saved to: {args.output.resolve()}", flush=True)
    return 1 if any(r["status"] == "error" for r in results) else 0


def main():
    """解析命令行参数并启动 Agent。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="baseline")
    p.add_argument(
        "--split",
        type=Path,
        help="Fixed reference/test split; all enhanced modes require it",
    )
    p.add_argument(
        "--reference-count",
        type=int,
        default=None,
        help="Default 4 fixed references; even counts >= 4 within balanced pool capacity",
    )
    p.add_argument("--dataset", type=Path, default=PROJECT / "data/unified")
    p.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Default: medical prompt for knowledge/combined, probability baseline otherwise",
    )
    p.add_argument("--provider", default="glm", help="Model adapter provider: glm or openai")
    p.add_argument(
        "--model",
        default=None,
        help="Model ID; provider default: glm-5.3-flash or gpt-5.6-terra",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=3,
        help="0 = all labeled images; default 3 for smoke run",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Auxiliary classification threshold; p_he >= threshold is positive",
    )
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"{type(e).__name__}: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
