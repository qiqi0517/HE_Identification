"""确定性形态路由：只用已校验标签，不看初判概率和测试真值。"""

PRIORITY = ("cystic", "mixed", "solid", "calcified", "no_focal_lesion")


def route(observation):
    """最多两分支；形态矛盾、看不清或无有效观察时使用通用分支。"""
    if observation is None or observation["quality"] == "uninterpretable":
        return ["general"]
    tags = set(observation["appearance_tags"])
    if "uncertain" in tags or ("no_focal_lesion" in tags and len(tags) > 1):
        return ["general"]
    branches = []
    for tag in PRIORITY:
        if tag in tags:
            branch = "mixed_solid" if tag in {"mixed", "solid"} else tag
            if branch not in branches:
                branches.append(branch)
    return branches[:2] or ["general"]
