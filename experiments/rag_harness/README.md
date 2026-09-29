# Agent 2.0：固定参考原图＋初筛—复核

> 历史原型：此目录仍为全样本两阶段实现。当前推荐使用主项目 `src/he_agent.py --review`，见[选择性复核指南](../../docs/experiment-guide.md#3-选择性复核agent-20)。以下内容用于追溯旧实验。


当前协议 `he-fixed-harness-v2`。TinyUSFM 已从运行流程移除，不需要编码器、权重、torch或索引。原历史索引/运行保留，不参与新推理。

```sh
# 从项目根目录运行；需要主Agent同样的Pillow环境
python experiments/rag_harness/run.py run --provider glm --mode review --reference-count 8 --limit 5 --dry-run
```

默认 `review`：固定8参考图、初筛—独立复核、无医学知识。`reference` 是同图单次调用对照；`knowledge_review` 显式使用原有知识卡。参考数量可选4/6/8，保持与v1相同的图片和顺序。

去掉 `--dry-run` 才会调用API，需要本地 `ZHIPU_API_KEY`；GPT 使用 `--provider openai`、`OPENAI_API_KEY`。旧 `rag`/`rag_harness`/`build-index`/`--index` 命令已停用，不静默重解释。

- [历史原型与实验摘要](../../docs/project-progress.md)
- [当前复核入口](../../docs/experiment-guide.md#3-选择性复核agent-20)
- [后续研究方案](../../docs/next-proposals.md)

测试：`python -m unittest discover -s experiments/rag_harness/tests -v`。
全量离线检查：`python experiments/rag_harness/scripts/verify_offline.py`，会阻止编码器依赖导入，验证两服务商三模式与两份划分的4/6/8参考。

此 README 只描述历史原型；当前主 Agent 已采用精简医学提示词和可选阴性复核，具体配置以主项目实验指南为准。
