"""选择并校验冻结提示词，让每次实验的医学知识与来源可追溯。

load_prompt 同时返回正文、版本指纹和来源记录；四模式由参考／知识两个开关组合。
冻结提示词与来源 JSON 必须一致，不能只改正文后沿用旧实验版本。
"""

import hashlib
import json
from pathlib import Path

from project_paths import load_paths

PROJECT = Path(__file__).resolve().parents[1]

# 一个模式由是否使用参考图、是否使用医学知识两个独立因素决定。
MODES = ("baseline", "reference", "knowledge", "reference_knowledge")
REFERENCE_MODES = {"reference", "reference_knowledge"}
KNOWLEDGE_MODES = {"knowledge", "reference_knowledge"}
MEDICAL_PROFILE = "medical-v4-concise-probability"


def load_prompt(mode, override=None, paths=None):
    """返回提示词、版本元数据及可选来源清单。

    知识模式同时校验医学提示词和基础提示词哈希，防止实验版本混用。
    """
    if mode not in MODES:
        raise ValueError(f"Unknown experiment mode: {mode}")
    paths = paths if paths is not None else load_paths(project=PROJECT)
    medical = paths["medical_prompt"]
    baseline = paths["baseline_prompt"]
    path = (
        Path(override) if override is not None else medical if mode in KNOWLEDGE_MODES else baseline
    )
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if not text.strip():
        raise ValueError("Empty prompt")
    digest = hashlib.sha256(raw).hexdigest()
    profile = "custom"
    evidence = None
    source_digest = None
    if path.resolve() == medical.resolve():
        source_bytes = paths["medical_sources"].read_bytes()
        evidence = json.loads(source_bytes)
        if evidence["prompt_sha256"] != digest:
            raise ValueError(
                "Frozen medical prompt changed; create a new version and update its evidence manifest"
            )
        if evidence["baseline_prompt_sha256"] != hashlib.sha256(baseline.read_bytes()).hexdigest():
            raise ValueError(
                "Baseline changed since the medical prompt was frozen; version prompts consistently"
            )
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        profile = evidence["version"]
    elif path.resolve() == baseline.resolve():
        profile = "baseline-probability-v2"
    if mode in KNOWLEDGE_MODES and profile != MEDICAL_PROFILE:
        raise ValueError(
            f"Knowledge modes require the frozen {MEDICAL_PROFILE} prompt; custom experiments can use baseline --prompt"
        )
    return (
        text,
        {
            "prompt_profile": profile,
            "prompt_path": str(path.resolve()),
            "prompt_file_sha256": digest,
            "knowledge_sources_sha256": source_digest,
        },
        evidence,
    )


def load_typing_prompt(mode, override=None, project=None):
    """分型提示始终保留输出契约；knowledge 模式追加冻结 PPT 摘要。"""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    root = Path(project) if project is not None else PROJECT
    base = Path(override) if override is not None else root / "prompts/he_typing.txt"
    prompt = base.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("Empty typing prompt")
    metadata = {"prompt_profile": "typing-confidence-v1", "baseline_path": str(base.resolve())}
    sources = None
    if mode in KNOWLEDGE_MODES:
        knowledge = root / "prompts/he_typing_knowledge.txt"
        source_path = root / "prompts/he_typing_knowledge.sources.json"
        sources = json.loads(source_path.read_text())
        raw = knowledge.read_bytes()
        if sources["knowledge_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Typing knowledge changed without source manifest update")
        default_base = root / "prompts/he_typing.txt"
        if sources["baseline_sha256"] != hashlib.sha256(default_base.read_bytes()).hexdigest():
            raise ValueError("Typing output contract changed without source manifest update")
        prompt += "\n" + raw.decode("utf-8")
        metadata["knowledge_sources_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        metadata["prompt_profile"] = "typing-confidence-knowledge-v1"
    metadata["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    return prompt, metadata, sources


def load_typing_review_prompt(mode, knowledge="checklist", override=None, base_override=None):
    """Build independent review input; checklist never inherits the initial full knowledge."""
    if knowledge not in ("same", "none", "checklist"):
        raise ValueError("Unknown review knowledge choice")
    review_mode = "knowledge" if knowledge == "same" and mode in KNOWLEDGE_MODES else "baseline"
    prompt, metadata, inherited_sources = load_typing_prompt(review_mode, base_override)
    path = Path(override) if override is not None else PROJECT / "prompts/he_typing_review.txt"
    supplement = path.read_text(encoding="utf-8")
    if not supplement.strip():
        raise ValueError("Empty review prompt")
    prompt += "\n" + supplement
    sources = inherited_sources
    if knowledge == "checklist":
        manifest = PROJECT / "prompts/he_typing_review.sources.json"
        sources = json.loads(manifest.read_text())
        for filename, expected in sources["files"].items():
            if (
                hashlib.sha256((PROJECT / "prompts" / filename).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Review checklist/contract changed without source manifest update")
        prompt += "\n" + (PROJECT / "prompts/he_typing_review_checklist.txt").read_text()
        metadata["knowledge_sources_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata.update(
        knowledge=knowledge,
        supplement_path=str(path.resolve()),
        supplement_sha256=hashlib.sha256(supplement.encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_profile="typing-review-" + knowledge,
    )
    return prompt, metadata, sources
