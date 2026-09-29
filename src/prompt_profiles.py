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
