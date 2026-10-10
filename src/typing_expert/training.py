"""Cached linear probes and limited finetuning with auditable checkpoints."""

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from . import CLASSES, SCHEMA
from .data import (
    code_hash,
    image_path,
    load_split,
    rows_for,
    sha,
    signature,
    verify_images,
    write,
    write_lines,
)
from .metrics import score
from .network import PREPROCESS, Expert, device_for, sync, tensor
from .options import class_weights, configure_head, resolve_options, stopping_key


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Images(Dataset):
    def __init__(self, root, rows, augment=False):
        self.root, self.rows, self.augment = root, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return tensor(image_path(self.root, row["image"]), self.augment), CLASSES.index(
            row["label"]
        )


def feature_cache(model, root, rows, device, cache, identity):
    """Verify source bytes even on cache hits; never accept a differently ordered matrix."""
    verify_images(root, rows)
    meta = dict(
        identity=identity,
        samples=[(r["image"], r["sha256"]) for r in rows],
        preprocessing=PREPROCESS,
        dimension=model.dimension,
    )
    key = signature(meta)
    path = Path(cache) / f"{key}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features = payload["features"]
        if payload["metadata"] != meta or features.shape != (len(rows), model.dimension):
            raise ValueError("Stale/invalid feature cache")
        if payload["tensor_sha256"] != tensor_hash(features) or not torch.isfinite(features).all():
            raise ValueError("Corrupt feature cache")
        return features
    model.encoder.eval()
    batches = []
    with torch.inference_mode():
        for x, _ in DataLoader(Images(root, rows), batch_size=16, shuffle=False):
            batches.append(model.encoder(x.to(device)).cpu())
    features = torch.cat(batches).float()
    if features.shape != (len(rows), model.dimension) or not torch.isfinite(features).all():
        raise ValueError("Invalid encoder features")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(metadata=meta, features=features, tensor_sha256=tensor_hash(features)), tmp)
    tmp.replace(path)
    return features


def tensor_hash(t):
    import hashlib

    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prediction(logits):
    logits = logits.detach().float().cpu()
    if logits.shape != (7,) or not torch.isfinite(logits).all():
        raise ValueError("Invalid logits")
    p = logits.softmax(0)
    index = int(p.argmax())
    return dict(
        status="ok",
        predicted_type=CLASSES[index],
        confidence=float(p[index]),
        confidence_kind="softmax_uncalibrated",
        probabilities=p.tolist(),
        logits=logits.tolist(),
    )


def checkpoint(model, config, epoch):
    return dict(
        schema=SCHEMA,
        classes=list(CLASSES),
        config=config,
        epoch=epoch,
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )


def load_checkpoint(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("classes") != list(CLASSES):
        raise ValueError("Checkpoint schema/class order mismatch")
    config = payload["config"]
    if config["preprocess"] != PREPROCESS or config["code_sha256"] != code_hash():
        raise ValueError(
            "Checkpoint implementation/preprocessing changed; use the saved code version"
        )
    model = Expert(config["variant"])
    configure_head(model, config.get("feature_scaling", "none"))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def validation(model, loader, rows, device, features=False):
    model.eval()
    values = []
    with torch.inference_mode():
        for x, _ in loader:
            values.append((model.head(x.to(device)) if features else model(x.to(device))).cpu())
    logits = torch.cat(values)
    predictions = [
        dict(image=r["image"], true_label=r["label"], **prediction(z)) for r, z in zip(rows, logits)
    ]
    metrics = score(rows, predictions)
    targets = torch.tensor([CLASSES.index(r["label"]) for r in rows])
    loss = float(nn.functional.cross_entropy(logits, targets))
    return predictions, metrics, loss


def train(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    options, default_options = resolve_options(args)
    split, root = load_split(args.split, args.dataset)
    train_rows, val_rows = rows_for(split, "train"), rows_for(split, "validation")
    if args.smoke:
        # Train-only smoke: no validation/test images, explicitly ineligible for selection.
        train_rows = [next(r for r in train_rows if r["label"] == c) for c in CLASSES]
        val_rows = train_rows
    verify_images(root, train_rows + val_rows)
    device = device_for(args.device)
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    model = Expert(args.variant, pretrained=True, tiny_weights=args.tiny_weights).to(device)
    configure_head(model, options["feature_scaling"])
    model.to(device)
    weights = class_weights(train_rows, options["class_weight"])
    if weights is not None:
        weights = weights.to(device)
    epochs = args.epochs or (30 if args.variant == "S3" else 50)
    if args.smoke:
        epochs = args.epochs or 2
    protocol = (
        not args.smoke and args.epochs is None and args.seed in (42, 43, 44) and default_options
    )
    config = dict(
        schema=SCHEMA,
        variant=args.variant,
        seed=args.seed,
        epochs=epochs,
        protocol_eligible=protocol,
        smoke=args.smoke,
        split_sha256=sha(args.split),
        code_sha256=code_hash(),
        dataset_hashes=split["dataset_hashes"],
        image_version=split["image_version"],
        preprocess=PREPROCESS,
        pretrained_sha256=model.pretrained_hash,
        device=str(device),
        batch_size=16,
        optimizer="AdamW",
        weight_decay=options["weight_decay"],
        loss="cross_entropy",
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        timm=__import__("timm").__version__,
        numpy=np.__version__,
        threads=args.threads,
        audit_status=split["audit_status"],
        learning_rates=dict(
            head=options["head_lr"],
            finetune_head=options["finetune_head_lr"],
            finetune_encoder=options["encoder_lr"],
        ),
        feature_scaling=options["feature_scaling"],
        class_weight=options["class_weight"],
        class_weights=weights.cpu().tolist() if weights is not None else None,
        early_stop_metric=options["early_stop_metric"],
        checkpoint_metric="accuracy_then_unweighted_validation_loss",
        warmup_epochs=5,
        patience=options["patience"],
    )
    output.mkdir(parents=True)
    write(output / "config.json", config)
    write(output / "split.json", split)
    write(output / "status.json", dict(status="running"))
    start = time.perf_counter()
    try:
        cached = args.variant in ("S1", "S2")
        if cached:
            identity = dict(
                pretrained=model.pretrained_hash,
                code=config["code_sha256"],
                encoder="tiny" if args.variant == "S1" else "resnet18",
            )
            datasets = [
                TensorDataset(
                    feature_cache(model, root, rows, device, args.cache, identity),
                    torch.tensor([CLASSES.index(r["label"]) for r in rows]),
                )
                for rows in (train_rows, val_rows)
            ]
        else:
            datasets = [Images(root, train_rows, augment=not args.smoke), Images(root, val_rows)]
        if options["feature_scaling"] == "standard":
            model.head[0].fit(datasets[0].tensors[0])
            write(
                output / "feature_scaling.json",
                dict(
                    fitted_role="train",
                    count=len(train_rows),
                    mean=model.head[0].mean.cpu().tolist(),
                    scale=model.head[0].scale.cpu().tolist(),
                ),
            )
        loader = DataLoader(datasets[0], batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(datasets[1], batch_size=16, shuffle=False, num_workers=0)

        def phase_groups(finetune=False):
            groups = model.phase(finetune)
            groups[0]["lr"] = options["finetune_head_lr"] if finetune else options["head_lr"]
            if finetune:
                groups[1]["lr"] = options["encoder_lr"]
            return groups

        optimizer = torch.optim.AdamW(phase_groups(), weight_decay=options["weight_decay"])
        predictions, metrics, loss = validation(model, val_loader, val_rows, device, cached)
        best_key, best_epoch, stale = (metrics["correct"], -loss), 0, 0
        torch.save(checkpoint(model, config, 0), output / "best.pt")
        stop_best = stopping_key(metrics["correct"], loss, options["early_stop_metric"])
        initial_loss = loss
        frozen_before = {k: tensor_hash(v) for k, v in model.encoder.state_dict().items()}
        history = [dict(epoch=0, validation_correct=metrics["correct"], validation_loss=loss)]
        for epoch in range(1, epochs + 1):
            finetune = args.variant == "S3" and epoch > 5
            groups = phase_groups(finetune)
            if args.variant == "S3" and epoch == 6:
                optimizer = torch.optim.AdamW(groups, weight_decay=options["weight_decay"])
                stale = 0
            total_loss, n = 0.0, 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.head(x) if cached else model(x)
                train_loss = nn.functional.cross_entropy(logits, y, weight=weights)
                if not torch.isfinite(train_loss):
                    raise ValueError("Non-finite training loss")
                train_loss.backward()
                optimizer.step()
                mass = float(weights[y].sum()) if weights is not None else len(y)
                total_loss += float(train_loss.detach()) * mass
                n += mass
            predictions, metrics, loss = validation(model, val_loader, val_rows, device, cached)
            key = (metrics["correct"], -loss)
            if key > best_key:
                best_key, best_epoch = key, epoch
                torch.save(checkpoint(model, config, epoch), output / "best.pt")
            stop_now = stopping_key(metrics["correct"], loss, options["early_stop_metric"])
            if stop_now > stop_best:
                stop_best, stale = stop_now, 0
            elif args.variant != "S3" or epoch > 5:
                stale += 1
            record = dict(
                epoch=epoch,
                train_loss=total_loss / n,
                validation_loss=loss,
                validation_correct=metrics["correct"],
                validation_accuracy=metrics["accuracy"],
                finetune=finetune,
                best_epoch=best_epoch,
                early_stop_stale=stale,
                learning_rates=[group["lr"] for group in optimizer.param_groups],
            )
            history.append(record)
            write_lines(output / "history.jsonl", history)
            print(json.dumps(dict(variant=args.variant, seed=args.seed, **record)), flush=True)
            if stale >= config["patience"]:
                break
        for key, before in frozen_before.items():
            should_freeze = cached or not (key.startswith(("blocks.10.", "blocks.11.", "norm.")))
            if should_freeze and tensor_hash(model.encoder.state_dict()[key]) != before:
                raise ValueError(f"Frozen encoder parameter/statistic changed: {key}")
        saved, payload = load_checkpoint(output / "best.pt", device)
        final_predictions, final_metrics, final_loss = validation(
            saved, val_loader, val_rows, device, cached
        )
        if final_metrics["correct"] != best_key[0] or not math.isclose(
            final_loss, -best_key[1], abs_tol=1e-6, rel_tol=1e-6
        ):
            raise ValueError("Checkpoint round-trip changed validation scores")
        write_lines(output / "validation_predictions.jsonl", final_predictions)
        write(output / "validation_metrics.json", final_metrics)
        majority = max(CLASSES, key=lambda c: sum(r["label"] == c for r in train_rows))
        baseline = [
            dict(image=r["image"], status="ok", predicted_type=majority, confidence=1.0)
            for r in val_rows
        ]
        write(
            output / "majority_baseline.json",
            dict(label=majority, metrics=score(val_rows, baseline)),
        )
        sync(device)
        summary = dict(
            status="complete",
            best_epoch=payload["epoch"],
            epochs_run=epoch,
            checkpoint_sha256=sha(output / "best.pt"),
            elapsed_seconds=time.perf_counter() - start,
            initial_validation_loss=initial_loss,
            best_validation_loss=final_loss,
            frozen_parameters_verified=True,
            checkpoint_roundtrip_verified=True,
            protocol_eligible=protocol,
        )
        write(output / "status.json", summary)
        return summary
    except BaseException as exc:
        write(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
