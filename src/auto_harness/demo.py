"""Offline synthetic end-to-end exercise of the actual typing workflow."""

import io
import json
from pathlib import Path

from PIL import Image

from models import ImagePart, ModelResponse
from typing_protocol import CLASSES

from . import search
from . import workflow as w
from .artifacts import file_hash, read_json
from .prompts import initial_bundle


class DemoVision:
    def __init__(self):
        self.requests = []

    def describe(self):
        return {"provider": "synthetic", "model": "typing-demo-vision"}

    def validate_credentials(self):
        pass

    def generate(self, request):
        self.requests.append(request)
        image = [x for x in request.content if isinstance(x, ImagePart)][-1]
        with Image.open(io.BytesIO(image.data)) as im:
            index, difficulty, _ = im.getpixel((0, 0))
        review = "独立、中性复核" in request.system_prompt
        pred = index if review or difficulty % 2 == 0 else (index + 1) % 7
        return ModelResponse(
            json.dumps(
                dict(
                    status="ok",
                    predicted_type=CLASSES[pred],
                    confidence=0.45 if difficulty % 2 else 0.9,
                    findings=["Synthetic visible structure."],
                )
            ),
            True,
            usage={"total_tokens": 10},
            finish_reason="stop",
        )


class DemoOptimizer:
    def describe(self):
        return {"provider": "synthetic", "model": "typing-demo-optimizer"}

    def validate_credentials(self):
        pass

    def generate(self, request):
        ctx = json.loads(request.content[0].text)
        extra = "observe_first" if ctx["round"] == 1 else "counterevidence"
        obj = dict(
            parent_version=ctx["parent"]["id"],
            evidence_ids=[ctx["parent"]["id"]],
            hypothesis="Synthetic prompt ordering experiment.",
            target_component="initial",
            prompt_patch={"initial_order": ["original", extra]},
            expected_effect="Unknown until measured.",
        )
        return ModelResponse(
            json.dumps(obj), True, usage={"total_tokens": 10}, finish_reason="stop"
        )


def initialize_demo(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Demo directory exists")
    (root / "data/images").mkdir(parents=True)
    split = dict(schema="he-typing-split-v1", partitions={}, samples={})
    for role in ("reference", "search", "selection", "test"):
        split["partitions"][role] = []
        for i, c in enumerate(CLASSES):
            name = f"images/{role}_{i}.png"
            path = root / "data" / name
            Image.new("RGB", (12, 12), (i, i % 2, 0)).save(path)
            split["partitions"][role].append(name)
            split["samples"][name] = dict(
                image=name,
                label=c,
                role=role,
                sha256=file_hash(path),
                group_id=f"{role}_{i}",
                source_dataset="synthetic",
                mime="image/png",
            )
    config = read_json(w.PROJECT / "configs/auto_harness.example.json")
    exp = dict(
        schema=w.SCHEMA,
        dataset=str(root / "data"),
        split=split,
        split_sha256="synthetic",
        config=config,
        clients=dict(vision=DemoVision().describe(), optimizer=DemoOptimizer().describe()),
        prompts=initial_bundle(w.PROJECT),
        engine_sha256=w.engine_hash(),
        simulation=True,
        protocol=dict(repetitions=3, rounds=4, stale_limit=2),
    )
    return w.freeze_experiment(root, exp)


def run_demo(root):
    root = Path(root)
    initialize_demo(root)
    client = DemoVision()
    w.collect(root, live=True, client=client)
    w.search(root)
    search.run(root, live=True, optimizer=DemoOptimizer(), vision=client)
    w.shortlist(root)
    w.collect(root, "selection", live=True, client=client)
    w.select(root)
    w.collect(root, "test", live=True, client=client)
    report = w.test(root)
    return dict(
        simulation=True,
        report=str(root / "test_report.json"),
        winner=report["winner"],
        scores={k: v["accuracy"] for k, v in report["scores"].items()},
    )
