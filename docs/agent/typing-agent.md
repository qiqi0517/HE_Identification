# 基础分型 Agent 运行指南

更新：2026-10-04。入口为 `python src/he_agent.py`，默认输出分型和 confidence。本流程只有初判和可选的一次中性复核，不运行 auto-harness 或任何优化器。

最新结果见 [五组实验首轮分析（2026-10-04）](typing-experiments-r1-2026-10-04.md)。

## 环境与已准备数据

在项目根目录、已激活的 Miniconda `he` 环境执行：

```bash
python -m pip install -r requirements-agent.txt
```

默认数据为 `data/unified_typing_v2/`，默认清单为 `configs/auto-harness-typing/split_v1.json`。两者已经准备好，无需再次生成。清单路径沿用方案约定的目录名，基础 Agent 只读取清单，不会启动 auto-harness。

| 集合 | 实际图片数 |
| --- | ---: |
| 参考 | 7，每个目标类别一张 |
| 搜索 | 190 |
| 选型 | 76 |
| 测试 | 119 |

新清单保留患者／来源关联、精确重复和保守 dHash 候选组，按类别及来源尽量分层。原目标 7／192／77／116 因保留整组调整为上表；没有拆组强求数量。医院与 Normal 缺患者 ID，图像分组不能保证患者独立；图内身份／诊断文字、临床标签与近重复人工审核仍待完成。

默认运行搜索集，避免开发时直接使用测试集。四种模式使用相同的集合、随机种子和样本顺序。`--limit 3` 为小规模运行，`--limit 0` 运行所选集合的全部图片。

## 先做本地校验

```bash
python src/he_agent.py \
  --mode reference_knowledge \
  --review --review-threshold 0.8 \
  --role search --limit 3 \
  --dry-run --output runs/typing_check
```

`--dry-run` 校验图片字节、分型标签、分组隔离、参考图和提示词；不读取密钥、不调用模型。输出目录必须不存在。

## 真实调用示例

```bash
python src/he_agent.py \
  --mode reference_knowledge \
  --review --review-threshold 0.8 \
  --role search --limit 3 \
  --output runs/typing_first_live
```

此命令会调用真实 API。默认服务商 GLM，读取已有 `ZHIPU_API_KEY` 环境变量或项目本地凭据文件；`--provider`、`--model`、`--max-tokens`、`--timeout`、`--retries` 保留。这里最多 6 次逻辑识别调用，传输重试可能增加实际请求数。

| 参数 | 含义 |
| --- | --- |
| `--mode baseline` | 仅待判图与基础任务提示 |
| `--mode reference` | 额外加入冻结参考图及其分型标签 |
| `--mode knowledge` | 额外加入固定分型医学知识，不输入参考图 |
| `--mode reference_knowledge` | 同时加入参考图和医学知识 |
| `--review` | 开启低 confidence 复核 |
| `--no-review` 或不写复核开关 | 仅初判 |
| `--review-threshold 0.8` | 有效初判 confidence 严格小于 0.8 时复核；等于阈值不复核 |
| `--reference-count N` | 参考模式取冻结顺序前 N 张；默认完整 7 张，取消旧偶数限制 |
| `--role search/selection/test` | 从对应集合取图，默认 search |
| `--dataset`、`--split` | 显式覆盖分型数据集与新格式清单 |
| `--prompt`、`--review-prompt` | 显式替换基础任务提示或中性复核补充提示；输出仍需符合协议 |

阈值 0.8 是可配置的初始规则，尚未优化或校准。NO_HE、CE1–CE5、AE 都按同一门控条件处理，不仅复核阴性；阈值为 0 时不触发复核，为 1 时 confidence 恰好 1 的图片不复核。初判弃权／错误不进入复核。旧 `--threshold` 患病概率阈值不再用于新入口。

## 输出和计分

初判与复核均要求：

```json
{
  "status": "ok",
  "predicted_type": "CE2",
  "confidence": 0.85,
  "findings": ["可见囊内子囊样结构"]
}
```

confidence 是本次类别正确性的自评，不是患病概率或经校准的正确率。弃权时使用 `status: abstain`，分型和 confidence 均为 null；无效字段、越界 confidence、截断响应记为错误。

复核不接收初判类别、confidence、理由或待判图真值。它使用同一原图，参考与知识由独立复核开关控制，默认同参考＋检查表；独立给出新的类别与自己的 confidence。最终结果完整替换初判，包含复核的弃权或错误；最多一次复核，不递归调用。

- `predictions.jsonl`：最终分型、confidence、状态、初判／复核完整记录及本地真值。
- `stages.jsonl`：每阶段结束立即保存，包括复核前已完成的初判。
- `metrics.json`：准确率、七类混淆、分类召回、来源统计、覆盖率，以及复核救回／引入错误和用量。
- `run.json`、`split.json`、`references.json`、`samples.json`：参数与输入快照。
- `prompt.txt`、可选 `review_prompt.txt`、`knowledge_sources.json`：实际提示和知识溯源。

主准确率分母是本次计划样本数；错误、弃权及致命故障后的未完成样本都计为不正确。测试标签只在本地评分和日志中使用，不加入模型请求。基础 Agent 每次运行每张图采集一次；研究方案的三次重复采集尚未实现。

## 提示词与实现

- 基础提示：`prompts/he_typing.txt`。
- 医学知识：`prompts/he_typing_knowledge.txt`，从用户 PPT 第 9–27 页整理；`he_typing_knowledge.sources.json` 保存内容哈希和来源。
- 中性复核：`prompts/he_typing_review.txt`。它保留不预设诊断方向的原则，并改为独立分型与 confidence。
- 当前流程：`src/typing_agent.py`、`src/typing_protocol.py`；`src/he_agent.py` 的 CLI 和 `run` 转入该流程。
- 历史概率函数保留供回归和旧依赖使用，旧程序调用 `run_probability`；历史 auto-harness 没有迁移到分型。

需要重新准备划分时，用不存在的新输出路径运行 `python src/typing_split.py --output 新清单路径.json`。该脚本不调用模型；现有固定划分不应在看过效果后重选。

分类专家小模型的训练与对照设计见 [分型专家小模型 v1 方案](../expert/typing-classifier-v1-plan.md)，已实现代码的完整命令见 [专家运行指南](../expert/typing-classifier-v1-runbook.md)。

当前数据已覆盖为 392 张，医院指定目录的 47 张作为 CE2 真值纳入。CL 禁止输出，解析时记为 error；无法判断仍可 abstain。新划分和提示须重新运行，历史结果不改写。复核输入建议见 [复核输入方案](review-input-proposal.md)，独立复核参考／知识开关已实现，完整命令见上述链接。

当前复核默认附加检查表。`--review-reference none|same` 控制复核参考；`--review-knowledge none|same|checklist` 控制复核知识，默认 checklist，不继承初判完整知识。推荐实验命令显式设置 0.6 阈值，CLI 默认仍为 0.8。
