"""Bounded outer prompt optimization with persisted evidence and resumable rounds."""

import logging
from pathlib import Path

from models import ModelRequest, TextPart

from . import workflow as w
from .artifacts import atomic_json, canonical, seal, unseal
from .policy import correct, route
from .prompts import SYSTEM, proposal


def events(root):
    return [unseal(p) for p in sorted((Path(root) / "optimizer/events").glob("*.json"))]


def state_for(root):
    history = events(root)
    incumbent, stale = "V0", 0
    for event in history:
        if event["accepted"]:
            incumbent, stale = event["version"], 0
        else:
            stale += 1
    pending = any(
        not (Path(root) / "optimizer/events" / p.name).exists()
        for p in (Path(root) / "optimizer/contexts").glob("*.json")
    )
    state = dict(incumbent=incumbent, stale=stale, rounds=len(history), pending=pending)
    atomic_json(Path(root) / "optimizer_state.json", state)
    return state


def summary(candidate):
    score = candidate["score"]
    return {
        k: score[k]
        for k in (
            "correct",
            "planned",
            "accuracy",
            "calls",
            "reviews",
            "rescued",
            "harmed",
            "repeats",
            "confidence_bins",
        )
    }


def context(root, exp, state):
    vid = state["incumbent"]
    ver = w.version(root, exp, vid)
    candidate = w.linked(root, f"search/{vid}/optimized.json", exp)["candidates"]["P_joint"]
    bank = w.bank_for(root, exp, vid)
    ids, cases, by_initial = [vid], [], {}
    for index, sample in enumerate(bank["samples"]):
        for rep, obs in enumerate(sample["repeats"]):
            initial = obs["initial"]
            action = route(candidate["policy"], initial)
            final = initial if action == "ACCEPT" else obs[action]
            key = initial["predicted_type"] if initial["status"] == "ok" else initial["status"]
            bucket = by_initial.setdefault(key, dict(count=0, correct=0, rescued=0, harmed=0))
            before, after = correct(initial, sample["label"]), correct(final, sample["label"])
            bucket["count"] += 1
            bucket["correct"] += after
            bucket["rescued"] += not before and after
            bucket["harmed"] += before and not after
            if not before or not after:
                eid = f"{vid}-case-{index}-{rep}"
                cases.append(
                    dict(
                        evidence_id=eid,
                        true_label=sample["label"],
                        action=action,
                        initial={
                            k: initial.get(k) for k in ("status", "predicted_type", "confidence")
                        },
                        outcomes={
                            a: {k: p.get(k) for k in ("status", "predicted_type", "confidence")}
                            for a, p in obs.items()
                        },
                    )
                )
    # No paths, patient/source IDs, image bytes or raw model findings leave this boundary.
    cases = cases[:20]
    ids += [c["evidence_id"] for c in cases]
    history = events(root)
    ids += [e["id"] for e in history]
    previous = []
    for event in history:
        previous.append(
            {
                k: event[k]
                for k in (
                    "id",
                    "parent_version",
                    "version",
                    "accepted",
                    "reason",
                    "proposal",
                    "score",
                    "diff",
                )
            }
        )
    # Source breakdown is deliberately omitted from optimizer context.
    score = summary(candidate)
    score["repeats"] = [{k: v for k, v in r.items() if k != "by_source"} for r in score["repeats"]]
    return dict(
        parent=ver,
        incumbent=dict(policy=candidate["policy"], score=score),
        history=previous,
        by_predicted_class=by_initial,
        cases=cases,
        evidence_ids=ids,
        approved_blocks={
            k: exp["prompts"][k] for k in ("initial_blocks", "review_blocks", "checklist_blocks")
        },
        scope="search_only",
        round=state["rounds"] + 1,
    )


def write_history(root):
    root = Path(root)
    history = events(root)
    path = root / "optimizer_history.jsonl"
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(canonical(e) + "\n" for e in history))
    temp.replace(path)
    atomic_json(
        root / "experience_summary.json",
        dict(
            experiments=[
                {k: e[k] for k in ("id", "accepted", "reason", "proposal", "score", "diff")}
                for e in history
            ],
            provenance="Generated from sealed experiment events; proposed hypotheses are not verified facts.",
        ),
    )
    state = state_for(root)
    exp = w.experiment(root)
    incumbent = w.linked(root, f"search/{state['incumbent']}/optimized.json", exp)["candidates"][
        "P_joint"
    ]
    atomic_json(
        root / "incumbent.json", dict(experiment_id=exp["experiment_id"], candidate=incumbent)
    )
    return state


def run(root, live=False, optimizer=None, vision=None):
    root, exp = Path(root), w.experiment(root)
    if (root / "shortlist.json").exists():
        raise ValueError("Search frozen; optimizer cannot access selection/test")
    w.linked(root, "search/V0/optimized.json", exp)
    state = state_for(root)
    if not live:
        return dict(
            dry_run=True,
            state=state,
            max_remaining_proposals=max(0, 4 - state["rounds"]),
            credentials_read=False,
        )
    while state["rounds"] < 4 and state["stale"] < 2:
        round_no = state["rounds"] + 1
        number = f"{round_no:02d}"
        cp = root / "optimizer/contexts" / f"{number}.json"
        ledger = w.Ledger(root, exp)
        if not cp.exists():
            # A new prompt version may invalidate all five observations per image.
            ledger.capacity("vision", len(w.rows_for(exp, "search")) * 15)
            ledger.capacity("optimizer")
            ctx = context(root, exp, state)
            seal(cp, dict(experiment_id=exp["experiment_id"], context=ctx, system=SYSTEM))
        else:
            ctx = w.linked(root, f"optimizer/contexts/{number}.json", exp)["context"]
        state_for(root)
        client = w.client_for(exp, "optimizer", optimizer)
        response = ledger.call(
            "optimizer",
            dict(
                experiment_id=exp["experiment_id"],
                round=round_no,
                system=SYSTEM,
                context=ctx,
                client=exp["clients"]["optimizer"],
            ),
            ModelRequest(SYSTEM, (TextPart(canonical(ctx)),)),
            client,
        )
        parent = ctx["parent"]
        event = dict(
            experiment_id=exp["experiment_id"],
            id=f"round-{round_no}",
            parent_version=parent["id"],
            version=None,
            accepted=False,
            reason=None,
            proposal=None,
            score=None,
            diff=None,
            request_id=response["request_id"],
        )
        try:
            if response.get("status") != "ok":
                raise ValueError("Optimizer response failed or incomplete")
            obj, spec = proposal(response["text"], parent, ctx["evidence_ids"], exp["prompts"])
            event["proposal"] = obj
            event["diff"] = {
                k: dict(before=parent["spec"][k], after=spec[k])
                for k in spec
                if spec[k] != parent["spec"][k]
            }
            if any(
                w.version(root, exp, p.stem)["spec"] == spec
                for p in (root / "prompt_versions").glob("*.json")
                if p.stem != f"V{round_no}"
            ):
                raise ValueError("Prompt already tried")
        except (ValueError, KeyError, TypeError) as exc:
            event["reason"] = f"rejected: {exc}"
        else:
            vid = f"V{round_no}"
            w.save_version(root, exp, vid, spec, parent["id"])
            w.collect(root, vid=vid, live=True, client=vision)
            candidate = w.search(root, vid)["candidates"]["P_joint"]
            previous = w.linked(root, f"search/{parent['id']}/optimized.json", exp)["candidates"][
                "P_joint"
            ]
            event.update(
                version=vid,
                score={
                    k: candidate["score"][k]
                    for k in ("correct", "planned", "accuracy", "calls", "rescued", "harmed")
                },
                accepted=w.rank_key(candidate) < w.rank_key(previous),
            )
            event["reason"] = (
                "improved_search_rank" if event["accepted"] else "no_search_rank_improvement"
            )
        seal(root / "optimizer/events" / f"{number}.json", event)
        logging.getLogger(__name__).info(
            "optimizer round %s: %s, accepted=%s", round_no, event["reason"], event["accepted"]
        )
        state = write_history(root)
    return state
