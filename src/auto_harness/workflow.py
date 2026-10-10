"""Seven-class acquisition, frozen stages, cache journals and deployment."""

import io
import logging
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from models import ModelError, create_model
from reference_split import validate_typing_split
from selective_review import predict_typing_stage, token_usage
from typing_agent import load_samples
from typing_protocol import CLASSES, build_request, parse_prediction

from .artifacts import atomic_json, file_hash, fingerprint, read_json, seal, unseal
from .evaluation import evaluate, paired, rank_key, replay, validate_bank
from .policy import ACTIONS, BASELINE, normalize, optimize, route
from .prompts import initial_bundle, render, validate_spec

SCHEMA = "he-typing-auto-harness-v1"
PROJECT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def engine_hash():
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [
        PROJECT / "src" / n
        for n in (
            "typing_protocol.py",
            "typing_agent.py",
            "selective_review.py",
            "reference_split.py",
        )
    ]
    files += sorted((PROJECT / "src/models").glob("*.py"))
    return fingerprint({str(p.relative_to(PROJECT)): file_hash(p) for p in files})


def create_client(settings):
    if set(settings) != {"provider", "model", "max_tokens", "timeout", "retries"}:
        raise ValueError("Unexpected model settings")
    if (
        not settings["model"]
        or type(settings["retries"]) is not int
        or settings["retries"] not in (0, 1)
    ):
        raise ValueError("Explicit model and at most one transport retry required")
    return create_model(**settings)


def initialize(root, split_path, dataset, config_path, project=PROJECT):
    root, dataset = Path(root), Path(dataset).resolve()
    if root.exists():
        raise FileExistsError("Use a new typing workspace; old binary workspaces are incompatible")
    split, config = read_json(split_path), read_json(config_path)
    samples = load_samples(dataset)
    _, refs = validate_typing_split(split, samples)
    if len(refs) != 7 or {r["label"] for r in refs} != set(CLASSES):
        raise ValueError("Exactly one reference per class required")
    hash_roles = {}
    for role, names in split["partitions"].items():
        for name in names:
            digest = split["samples"][name]["sha256"]
            if hash_roles.setdefault(digest, role) != role:
                raise ValueError("Exact image duplicate crosses partitions")
    if set(config) != {"vision", "optimizer", "budgets"}:
        raise ValueError("Use the typing auto_harness.example.json configuration")
    budgets = config["budgets"]
    if set(budgets) != {
        "vision_calls",
        "vision_reported_tokens",
        "optimizer_calls",
        "optimizer_reported_tokens",
    } or any(type(v) is not int or v <= 0 for v in budgets.values()):
        raise ValueError("Positive integer budgets required")
    if budgets["optimizer_calls"] > 4:
        raise ValueError("At most four optimizer proposals")
    clients = {k: create_client(config[k]).describe() for k in ("vision", "optimizer")}
    for sample in samples:
        split["samples"][sample["image"]].update(sample)
    exp = dict(
        schema=SCHEMA,
        dataset=str(dataset),
        split=split,
        split_sha256=file_hash(split_path),
        config=config,
        clients=clients,
        dataset_hashes={
            n: file_hash(dataset / "labels" / n) for n in ("labels_typing.txt", "manifest.jsonl")
        },
        prompts=initial_bundle(project),
        engine_sha256=engine_hash(),
        simulation=False,
        protocol=dict(repetitions=3, rounds=4, stale_limit=2, actions=list(ACTIONS)),
    )
    root.mkdir(parents=True)
    return freeze_experiment(root, exp)


def freeze_experiment(root, exp):
    exp["experiment_id"] = fingerprint(exp)
    seal(Path(root) / "experiment.json", exp)
    atomic_json(Path(root) / "budget.json", dict(limits=exp["config"]["budgets"], revisions=[]))
    save_version(root, exp, "V0", exp["prompts"]["spec"], None)
    return exp


def experiment(root):
    exp = unseal(Path(root) / "experiment.json")
    if exp.get("schema") != SCHEMA:
        raise ValueError(
            "Old binary auto-harness workspace is incompatible; initialize a new typing workspace"
        )
    if fingerprint({k: v for k, v in exp.items() if k != "experiment_id"}) != exp["experiment_id"]:
        raise ValueError("Experiment identity mismatch")
    if exp["engine_sha256"] != engine_hash():
        raise ValueError("Code changed after freeze; initialize a new workspace")
    if not exp["simulation"]:
        for name, digest in exp["dataset_hashes"].items():
            if file_hash(Path(exp["dataset"]) / "labels" / name) != digest:
                raise ValueError("Dataset labels/manifest changed after freeze")
    return exp


def linked(root, name, exp):
    value = unseal(Path(root) / name)
    if value.get("experiment_id") != exp["experiment_id"]:
        raise ValueError("Artifact belongs to another experiment")
    return value


def save_version(root, exp, vid, spec, parent):
    if not vid.startswith("V") or not vid[1:].isdigit() or not 0 <= int(vid[1:]) <= 4:
        raise ValueError("Version must be V0..V4")
    value = dict(
        experiment_id=exp["experiment_id"],
        id=vid,
        parent=parent,
        spec=validate_spec(exp["prompts"], spec),
        prompts=render(exp["prompts"], spec),
    )
    path = Path(root) / "prompt_versions" / f"{vid}.json"
    if path.exists():
        if unseal(path) != value:
            raise ValueError("Prompt version already frozen differently")
        return value
    return seal(path, value)


def version(root, exp, vid):
    if vid not in ("V0", "V1", "V2", "V3", "V4"):
        raise ValueError("Invalid version")
    value = linked(root, f"prompt_versions/{vid}.json", exp)
    if value["prompts"] != render(exp["prompts"], value["spec"]):
        raise ValueError("Prompt rendering mismatch")
    return value


def rows_for(exp, role):
    return [{**exp["split"]["samples"][n], "image": n} for n in exp["split"]["partitions"][role]]


def image_data(exp, name):
    base = Path(exp["dataset"])
    path = (base / name).resolve()
    if (
        not path.is_relative_to(base / "images")
        or file_hash(path) != exp["split"]["samples"][name]["sha256"]
    ):
        raise ValueError("Image path/hash changed")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[im.format]
    return data, mime


def references(exp):
    result = []
    for n in exp["split"]["partitions"]["reference"]:
        data, mime = image_data(exp, n)
        result.append(dict(bytes=data, mime=mime, label=exp["split"]["samples"][n]["label"]))
    return result


def error_result(message):
    return dict(
        status="error",
        predicted_type=None,
        confidence=None,
        findings=[],
        error=message,
        usage=None,
        latency_seconds=0,
    )


class Ledger:
    """Reserve before sending; interrupted requests never silently re-bill on resume."""

    def __init__(self, root, exp):
        self.root, self.exp = Path(root), exp
        self.limits = read_json(self.root / "budget.json")["limits"]
        self.entries = {}
        self.counts = {"vision": 0, "optimizer": 0}
        self.tokens = {"vision": 0, "optimizer": 0}
        for path in sorted((self.root / "requests/reserved").glob("*.json")):
            entry = unseal(path)
            if entry["experiment_id"] != exp["experiment_id"] or path.stem != fingerprint(
                entry["identity"]
            ):
                raise ValueError("Request journal binding mismatch")
            done = self.root / "requests/completed" / path.name
            if done.exists():
                result = unseal(done)
                if result["request_id"] != path.stem:
                    raise ValueError("Response journal binding mismatch")
            else:
                result = dict(
                    request_id=path.stem, result=error_result("interrupted_request_billing_unknown")
                )
                seal(done, result)
            self.entries[path.stem] = (entry, result["result"])
            self.account(entry["kind"], result["result"])

    def account(self, kind, result):
        self.counts[kind] += 1
        self.tokens[kind] += token_usage([result])["total_tokens"] or 0

    def usage(self, kind):
        items = [r for e, r in self.entries.values() if e["kind"] == kind]
        return dict(calls=len(items), **{"usage": token_usage(items)})

    def capacity(self, kind, calls=1):
        if self.counts[kind] + calls > self.limits[kind + "_calls"]:
            raise ValueError(
                f"{kind} call budget reached; cached progress retained. Use budget command to raise limit."
            )
        tokens = self.tokens[kind]
        if tokens >= self.limits[kind + "_reported_tokens"]:
            raise ValueError(
                f"{kind} reported-token stop reached; cached progress retained. Use budget command to raise limit."
            )

    def call(self, kind, identity, request, client):
        key = fingerprint(identity)
        if key in self.entries:
            return {**self.entries[key][1], "request_id": key}
        self.capacity(kind)
        entry = dict(experiment_id=self.exp["experiment_id"], identity=identity, kind=kind)
        seal(self.root / "requests/reserved" / f"{key}.json", entry)
        if kind == "vision":
            result = predict_typing_stage(client, request, parse_prediction)
        else:
            started = time.monotonic()
            try:
                response = client.generate(request)
                result = {
                    **asdict(response),
                    "status": "ok" if response.completed else "error",
                    "latency_seconds": time.monotonic() - started,
                }
            except ModelError as exc:
                result = {**error_result(str(exc)), "fatal": exc.fatal}
        seal(self.root / "requests/completed" / f"{key}.json", dict(request_id=key, result=result))
        self.entries[key] = (entry, result)
        self.account(kind, result)
        if result.get("fatal"):
            raise ValueError(
                "Fatal model response saved; fix credentials/service before resuming. Cached failure remains counted."
            )
        return {**result, "request_id": key}


def client_for(exp, kind, supplied=None):
    client = supplied or create_client(exp["config"][kind])
    if client.describe() != exp["clients"][kind]:
        raise ValueError("Client differs from frozen model settings")
    client.validate_credentials()
    return client


def observation(exp, ver, role, sample, rep, stage, client, ledger, refs):
    use_refs = stage in ("REVIEW_REFERENCE", "REVIEW_REFERENCE_CHECKLIST")
    prompt = ver["prompts"][stage]
    identity = dict(
        experiment_id=exp["experiment_id"],
        role=role,
        image=sample["image"],
        image_sha256=sample["sha256"],
        rep=rep,
        prompt=prompt,
        references=[(r["sha256"], r["label"]) for r in rows_for(exp, "reference")]
        if use_refs
        else [],
        client=exp["clients"]["vision"],
    )
    data, mime = image_data(exp, sample["image"])
    request = build_request(data, mime, prompt, refs if use_refs else ())
    return ledger.call("vision", identity, request, client)


def bank_for(root, exp, vid):
    bank = linked(root, f"search/{vid}/bank.json", exp)
    ver = version(root, exp, vid)
    if bank["version_sha256"] != fingerprint(ver):
        raise ValueError("Bank prompt binding mismatch")
    validate_bank(bank)
    expected = rows_for(exp, "search")
    if [{k: v for k, v in s.items() if k != "repeats"} for s in bank["samples"]] != expected:
        raise ValueError("Bank sample metadata mismatch")
    return bank


def collect(root, role="search", vid="V0", live=False, client=None):
    root, exp = Path(root), experiment(root)
    if role != "search":
        return collect_candidates(root, role, live, client)
    if (root / "shortlist.json").exists():
        raise ValueError("Search is frozen")
    ver = version(root, exp, vid)
    path = root / f"search/{vid}/bank.json"
    if path.exists():
        return bank_for(root, exp, vid)
    samples = rows_for(exp, "search")
    if not live:
        return dict(
            dry_run=True,
            images=len(samples),
            repetitions=3,
            max_logical_calls=len(samples) * 15,
            version=vid,
            credentials_read=False,
        )
    ledger = Ledger(root, exp)
    # On a fresh version, reserve room for its full worst-case collection; resume uses journals.
    if not any(e["identity"].get("role") == "search" for e, r in ledger.entries.values()):
        ledger.capacity("vision", len(samples) * 15)
    client = client_for(exp, "vision", client)
    refs = references(exp)
    bank = dict(
        schema="he-typing-action-bank-v1",
        experiment_id=exp["experiment_id"],
        version_sha256=fingerprint(ver),
        role="search",
        repetitions=3,
        samples=[],
    )
    for sample in samples:
        obs = []
        for rep in range(3):
            initial = observation(exp, ver, "search", sample, rep, "initial", client, ledger, refs)
            stages = dict(initial=initial)
            if initial["status"] == "ok":
                for action in ACTIONS[1:]:
                    stages[action] = observation(
                        exp, ver, "search", sample, rep, action, client, ledger, refs
                    )
            obs.append(stages)
        bank["samples"].append({**sample, "repeats": obs})
        LOG.info(
            "search %s: %s/%s images complete; %s unique vision calls saved",
            vid,
            len(bank["samples"]),
            len(samples),
            ledger.counts["vision"],
        )
    validate_bank(bank)
    return seal(path, bank)


def search(root, vid="V0"):
    root, exp = Path(root), experiment(root)
    if (root / "shortlist.json").exists():
        raise ValueError("Search is frozen")
    path = root / f"search/{vid}/optimized.json"
    if path.exists():
        return linked(root, f"search/{vid}/optimized.json", exp)
    bank, ver = bank_for(root, exp, vid), version(root, exp, vid)
    candidates = {}
    for name, segmented in (("P_joint", True), ("P_type", False)):
        policy, scores = optimize(bank, segmented)
        rule_path = root / f"search/{vid}/{name}_rules.json"
        rule_data = dict(experiment_id=exp["experiment_id"], scores=scores)
        if rule_path.exists():
            if unseal(rule_path) != rule_data:
                raise ValueError("Partial optimization rules differ from frozen bank")
        else:
            seal(rule_path, rule_data)
        candidates[name] = dict(
            id=vid,
            spec=ver["spec"],
            policy=policy,
            score=evaluate(replay(bank, policy), bank["samples"]),
        )
    return seal(
        path,
        dict(
            experiment_id=exp["experiment_id"],
            version=vid,
            bank_sha256=fingerprint(bank),
            candidates=candidates,
        ),
    )


def shortlist(root):
    root, exp = Path(root), experiment(root)
    if (root / "shortlist.json").exists():
        return linked(root, "shortlist.json", exp)
    from .search import state_for

    state_for(root)
    base = linked(root, "search/V0/optimized.json", exp)["candidates"]["P_joint"]
    if (root / "optimizer_state.json").exists():
        state = read_json(root / "optimizer_state.json")
        if state.get("pending"):
            raise ValueError("Finish pending optimizer round before freezing")
        dynamic = linked(root, f"search/{state['incumbent']}/optimized.json", exp)["candidates"][
            "P_joint"
        ]
    else:
        dynamic = base
    raw = [("S0", {**base, "policy": BASELINE}), ("P_static", base), ("P_dynamic", dynamic)]
    candidates = []
    for alias, c in raw:
        match = next(
            (r for r in candidates if (r["spec"], r["policy"]) == (c["spec"], c["policy"])), None
        )
        if match:
            match["aliases"].append(alias)
        else:
            candidates.append(
                dict(
                    name=alias, aliases=[alias], version=c["id"], spec=c["spec"], policy=c["policy"]
                )
            )
    return seal(
        root / "shortlist.json", dict(experiment_id=exp["experiment_id"], candidates=candidates)
    )


def collect_candidates(root, role, live=False, client=None):
    root, exp = Path(root), experiment(root)
    if role not in ("selection", "test"):
        raise ValueError("Invalid role")
    frozen = linked(root, "shortlist.json", exp)
    if role == "test":
        linked(root, "selected.json", exp)
    target = root / role / "results.json"
    if target.exists():
        return linked(root, f"{role}/results.json", exp)
    if role == "selection" and (root / "selected.json").exists():
        raise ValueError("Selection frozen")
    samples = rows_for(exp, role)
    if not live:
        return dict(
            dry_run=True,
            role=role,
            images=len(samples),
            credentials_read=False,
            max_logical_calls=len(samples)
            * 3
            * sum(1 if "S0" in c["aliases"] else 2 for c in frozen["candidates"]),
        )
    ledger, refs = Ledger(root, exp), references(exp)
    client = client_for(exp, "vision", client)
    results = {}
    for c in frozen["candidates"]:
        ver, policy = version(root, exp, c["version"]), normalize(c["policy"])
        rows = []
        for sample in samples:
            for rep in range(3):
                initial = observation(exp, ver, role, sample, rep, "initial", client, ledger, refs)
                action = route(policy, initial)
                final = (
                    initial
                    if action == "ACCEPT"
                    else observation(exp, ver, role, sample, rep, action, client, ledger, refs)
                )
                rows.append(
                    dict(
                        image=sample["image"], rep=rep, initial=initial, action=action, final=final
                    )
                )
            LOG.info("%s %s: %s/%s images complete", role, c["name"], len(rows) // 3, len(samples))
        results[c["name"]] = dict(rows=rows, score=evaluate(rows, samples))
    return seal(
        target,
        dict(
            experiment_id=exp["experiment_id"],
            shortlist_sha256=fingerprint(frozen),
            results=results,
        ),
    )


def checked_results(root, exp, role):
    frozen = linked(root, "shortlist.json", exp)
    data = linked(root, f"{role}/results.json", exp)
    if data["shortlist_sha256"] != fingerprint(frozen) or set(data["results"]) != {
        c["name"] for c in frozen["candidates"]
    }:
        raise ValueError("Candidate result binding mismatch")
    for c in frozen["candidates"]:
        result = data["results"][c["name"]]
        for row in result["rows"]:
            if row["action"] != route(c["policy"], row["initial"]):
                raise ValueError("Executed route differs from frozen policy")
        if result["score"] != evaluate(result["rows"], rows_for(exp, role)):
            raise ValueError("Stored score mismatch")
    return frozen, data


def select(root):
    root, exp = Path(root), experiment(root)
    if (root / "selected.json").exists():
        return linked(root, "selected.json", exp)
    frozen, data = checked_results(root, exp, "selection")
    candidates = [{**c, "score": data["results"][c["name"]]["score"]} for c in frozen["candidates"]]
    best = min(candidates, key=rank_key)
    return seal(
        root / "selected.json",
        dict(
            experiment_id=exp["experiment_id"],
            winner=best,
            selection_sha256=fingerprint(data),
            shortlist_sha256=fingerprint(frozen),
        ),
    )


def test(root):
    root, exp = Path(root), experiment(root)
    selected = linked(root, "selected.json", exp)
    if (root / "test_report.json").exists():
        return linked(root, "test_report.json", exp)
    frozen, data = checked_results(root, exp, "test")
    baseline = next(c["name"] for c in frozen["candidates"] if "S0" in c["aliases"])
    comparisons = {
        c["name"]: paired(
            data["results"][c["name"]]["rows"],
            data["results"][baseline]["rows"],
            rows_for(exp, "test"),
        )
        for c in frozen["candidates"]
        if c["name"] != baseline
    }
    ledger = Ledger(root, exp)
    return seal(
        root / "test_report.json",
        dict(
            experiment_id=exp["experiment_id"],
            simulation=exp["simulation"],
            winner=selected["winner"]["name"],
            scores={k: v["score"] for k, v in data["results"].items()},
            paired_vs_baseline=comparisons,
            test_sha256=fingerprint(data),
            acquisition_usage={kind: ledger.usage(kind) for kind in ("vision", "optimizer")},
        ),
    )


def export(root, output):
    exp = experiment(root)
    if exp["simulation"]:
        raise ValueError("Synthetic experiments cannot be deployed")
    winner = linked(root, "selected.json", exp)["winner"]
    ver = version(root, exp, winner["version"])
    refs = exp["split"]["partitions"]["reference"]
    deploy = dict(
        schema="he-typing-harness-deployment-v1",
        engine_sha256=exp["engine_sha256"],
        dataset=exp["dataset"],
        client=exp["clients"]["vision"],
        settings=exp["config"]["vision"],
        prompts=ver["prompts"],
        policy=winner["policy"],
        experiment_id=exp["experiment_id"],
        references=[
            {k: exp["split"]["samples"][n][k] for k in ("image", "label", "sha256")} for n in refs
        ],
    )
    return seal(output, deploy)


def predict(deployment, image, live=False, client=None):
    dep = unseal(deployment)
    if (
        dep.get("schema") != "he-typing-harness-deployment-v1"
        or dep["engine_sha256"] != engine_hash()
    ):
        raise ValueError("Incompatible deployment")
    policy = normalize(dep["policy"])
    data = Path(image).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[im.format]
    refs = []
    for r in dep["references"]:
        path = (Path(dep["dataset"]) / r["image"]).resolve()
        if (
            not path.is_relative_to(Path(dep["dataset"]) / "images")
            or file_hash(path) != r["sha256"]
        ):
            raise ValueError("Deployment reference changed")
        with Image.open(path) as im:
            ref_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[im.format]
        refs.append(dict(bytes=path.read_bytes(), mime=ref_mime, label=r["label"]))
    if not live:
        return dict(dry_run=True, max_logical_calls=2, credentials_read=False)
    client = client or create_client(dep["settings"])
    if client.describe() != dep["client"]:
        raise ValueError("Client differs from deployment")
    client.validate_credentials()
    initial = predict_typing_stage(
        client, build_request(data, mime, dep["prompts"]["initial"]), parse_prediction
    )
    action = route(policy, initial)
    final = initial
    if action != "ACCEPT":
        final = predict_typing_stage(
            client,
            build_request(
                data,
                mime,
                dep["prompts"][action],
                refs if action in ("REVIEW_REFERENCE", "REVIEW_REFERENCE_CHECKLIST") else (),
            ),
            parse_prediction,
        )
    return dict(
        initial=initial, action=action, final=final, logical_calls=1 + int(action != "ACCEPT")
    )


def update_budget(root, changes):
    root = Path(root)
    experiment(root)
    value = read_json(root / "budget.json")
    if (
        not changes
        or not set(changes) <= set(value["limits"])
        or any(type(v) is not int or v < value["limits"][k] for k, v in changes.items())
    ):
        raise ValueError("Budget updates may only increase existing positive limits")
    if changes.get("optimizer_calls", 0) > 4:
        raise ValueError("Four proposal cap cannot be increased")
    value["revisions"].append(dict(before=value["limits"].copy(), changes=changes))
    value["limits"].update(changes)
    atomic_json(root / "budget.json", value)
    return value
