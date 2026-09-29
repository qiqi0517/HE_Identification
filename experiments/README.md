# 历史实验区

当前可运行主 Agent 在 `src/he_agent.py`。此处保留核心尝试及复现代码，不参与主流程，也不把离线跑通当作诊断效果。

| 子项目 | 保留价值 | 当前结论 |
| --- | --- | --- |
| [expert_lab](expert_lab/README.md) | 权重核验、TinyUSFM 特征／相似检索、疾病 checkpoint 探针 | 无已验证的 HE 概率专家；不接入主 Agent |
| [rag_harness](rag_harness/README.md) | 初筛、知识卡和两阶段调用的原型 | 主流程已采用选择性阴性复核；这里供历史追溯 |

这些子项目的依赖、权重及生成结果各自管理。源码中新增说明不改变原实验算法；第三方 `research/` 摘录、权重和历史结果保持原样。统一的研究结论见[项目进展](../docs/project-progress.md)，未来专家／检索设计见[研究提案](../docs/next-proposals.md)。
