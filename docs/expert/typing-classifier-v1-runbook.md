# 分型专家小模型 v1：运行指南

更新：2026-10-04。代码已实现 S1/S2/S3 训练、验证选型、冻结测试、单图预测及配对比较。设计依据见 [训练方案](typing-classifier-v1-plan.md)。这是独立分类专家，不调用 LLM，也尚未接入基础 Agent 或 Auto Harness。

## 1. 在 he 环境安装依赖

从项目根目录执行：

```bash
conda activate he
cd /Users/qiqi/Research/HE_Identification
python -m pip install -r requirements-expert.txt
```

本次检查发现 `he` 是 Python 3.11.16，已有 NumPy/Pillow，但尚无 torch、torchvision、timm。没有替用户修改该环境；本轮验证使用仓库已有 `experiments/expert_lab/.venv`。依赖固定在该环境已验证的组合。

默认 `--device auto` 按 CUDA、MPS、CPU 顺序选择可用设备，可显式改为 `--device cpu`。本轮只验证 CPU，尚未验证本机 MPS/CUDA 执行。S1/S3 使用已经存在的 `experiments/expert_lab/weights/TinyUSFM.pth` 并严格校验哈希；S2 首次运行会由 torchvision 下载 ImageNet ResNet18 权重到 PyTorch 缓存，需联网，之后可离线复用。单图预测的完整检查点自带编码器，不再下载预训练权重。

## 2. 数据映射（本次已生成，可跳过）

已生成 `configs/expert-typing/v1/split.json`，train 197、validation 76、test 119。它保留原参考／搜索／选型／测试清单，只映射训练用途。

若在新克隆中还没有此文件，执行：

```bash
python scripts/experiments/train_typing_expert.py prepare
```

输出文件存在时默认拒绝覆盖；本次按用户要求已覆盖，今后可显式添加 `--overwrite` 更新原路径。需要重新整理数据时用新的 `--output` 路径；后续 train 使用相应 `--split`。prepare 读取标签和元数据，不提取或读取测试图像像素；训练只读取训练／验证图片。数据哈希或关联组不一致会停止。

当前映射记录 `audit_status=patient/text/clinical/pretraining_overlap_review_pending`。本轮没有替代患者关联、临床标签及图内文字人工审核，图片仍为当前原始版本，因此正式训练结果仍属内部探索。若人工审核后生成派生图像和新版标签／清单，应另建实验版本，并让 LLM 对照使用同版图像。

## 3. 可选的小规模检查

```bash
python scripts/experiments/train_typing_expert.py train \
  --variant S1 --seed 42 --smoke --epochs 8 \
  --output runs/expert_typing/my_smoke_s1
```

只取 train 每类第一张，共 7 张，训练和检查均使用这 7 张，不读取 validation/test。此命令验证权重、梯度、保存／加载，不衡量泛化；`protocol_eligible=false`，不能参与选型。需要检查 S3 第 6 轮解冻时，将配置改为 `--variant S3 --epochs 6`，换新输出目录。

## 4. 正式训练：先跑 S1，再完成九次训练

可以先用这一条确认正式训练能完成：

```bash
python scripts/experiments/train_typing_expert.py train \
  --variant S1 --seed 42 \
  --output runs/expert_typing/v1/S1/42
```

读取全部 197 张训练图和 76 张验证图。S1 只训练线性头，最多 50 轮，按验证正确数、验证交叉熵选检查点，连续 10 轮未改善即早停。

如果还没有运行任何正式训练，可直接执行下面完整批次；若已经运行上面 S1/42，不要重复运行同目录，手动跳过该组合，再运行余下 8 个组合：

```bash
for variant in S1 S2 S3; do
  for seed in 42 43 44; do
    python scripts/experiments/train_typing_expert.py train \
      --variant "$variant" --seed "$seed" \
      --output "runs/expert_typing/v1/$variant/$seed" || break 2
  done
done
```

S1/S2 共享各自的确定性原始特征缓存；S3 重新读取图片并做轻微增强，前 5 轮训练头，第 6 轮起只微调最后两个 block 和末端 norm，最多 30 轮。缓存校验包含原图哈希、编码器、处理版本及代码指纹，不使用旧 L2 归一化检索缓存。

正式 v1 命令不要加 `--smoke` 或 `--epochs`：自定义轮数会标为非标准实验，即使数值恰好等于默认值也不能混入 v1 选型。所有输出目录必须不存在，现阶段不支持断点续训；失败后保留失败目录，换新批次根目录，或将失败目录归档后重跑该组合。不要混合不同代码／数据版本的运行。

## 5. 验证选型并冻结

九次训练均成功后执行：

```bash
python scripts/experiments/train_typing_expert.py select \
  --runs runs/expert_typing/v1 \
  --output runs/expert_typing/v1/selection.json
```

检查九个运行是否完整且使用相同划分／实现，复算验证指标，按三个种子的正确数之和选配置；并列依次比较平均宏召回率和 S1/S2/S3 顺序。部署检查点固定选胜出配置的 seed=42。冻结文件列出全部候选与权重哈希，不根据测试表现改选。

## 6. 测试冻结的模型

```bash
python scripts/experiments/train_typing_expert.py evaluate \
  --freeze runs/expert_typing/v1/selection.json \
  --output runs/expert_typing/v1/test_selected
```

此时才读取 119 张 test 图片。报告准确率、逐类召回率、macro-F1、宏召回率、混淆矩阵、来源／医院／去 Normal 来源子集、confidence 分箱、有效概率输出的 NLL/Brier，以及 batch=1 耗时和进程峰值内存。读图或推理错误计入完整分母；哈希变更导致实验停止并标记失败，已写入指标保留未完成分母。NLL/Brier 仅描述成功概率输出，并显式保存样本数，不用于掩盖错误。

若需要方案预设的全部种子对照，先冻结选型，再执行：

```bash
for variant in S1 S2 S3; do
  for seed in 42 43 44; do
    python scripts/experiments/train_typing_expert.py evaluate \
      --freeze runs/expert_typing/v1/selection.json \
      --variant "$variant" --seed "$seed" \
      --output "runs/expert_typing/v1/test_${variant}_${seed}" || break 2
  done
done
```

`evaluate` 不重新优化模型。新测试分数只能用于报告，不能据此改变 `selection.json`。同一检查点确定性推理无需重复执行；`test_selected` 与其对应的 `test_Sx_42` 功能相同，可二选一保留。

## 7. 预测单张新图

把下面检查点替换为 `selection.json` 的 `selected.checkpoint`，图片路径替换为实际文件：

```bash
python scripts/experiments/train_typing_expert.py predict \
  --checkpoint runs/expert_typing/v1/S1/42/best.pt \
  --image /绝对路径/待识别图片.jpg
```

输出包括 `predicted_type`、`confidence`、七类 `probabilities`、`logits`、模型版本与权重哈希。七类向量顺序固定为 `NO_HE, CE1, CE2, CE3, CE4, CE5, AE`；confidence 是未校准 softmax 最大值。它不生成 findings，也不会把低置信度直接当成 CL。

## 8. 与 LLM 进行同图测试比较

LLM 使用已有 [基础 Agent](../agent/typing-agent.md) CLI。模型、提示和输入版本应在测试前冻结；以下会实际调用 API，需已有密钥，且会产生费用。本次代码验证未执行这些调用。

```bash
for repeat in 1 2 3; do
  python src/he_agent.py --provider glm --model glm-5.3-flash \
    --mode baseline --no-review --role test --limit 0 --seed 42 \
    --output "runs/expert_typing/llm_test/e1_r${repeat}" || break
  python src/he_agent.py --provider glm --model glm-5.3-flash \
    --mode reference --reference-count 7 --no-review --role test --limit 0 --seed 42 \
    --output "runs/expert_typing/llm_test/e2_r${repeat}" || break
done
```

对一次同图结果进行配对比较：

```bash
python scripts/experiments/train_typing_expert.py compare \
  --first runs/expert_typing/v1/test_selected \
  --second runs/expert_typing/llm_test/e1_r1 \
  --output runs/expert_typing/v1/compare_e1_r1.json
```

比较入口检查测试角色、图片集合、标签和原图哈希一致；输出两方完整指标及按已知关联组的 20,000 次配对 bootstrap，种子 42。第一方必须为专家测试运行，提供关联组。三次 LLM 输出分别比较、汇总，不能把重复运行当成新患者，也不能与之前 167 张 search 成绩直接比较。

## 9. 产物与实现验证

训练目录包含 `config.json`、`split.json`、`history.jsonl`、`best.pt`、`validation_predictions.jsonl`、`validation_metrics.json`、`majority_baseline.json` 和 `status.json`。`best.pt` 保存完整编码器、分类头、类别顺序和配置，采用只加载张量／基础容器的 `weights_only=True`。选型／推理校验代码指纹；修改专家实现后应另开实验版本，或恢复原实现重放旧运行。

本轮验证使用真实 TinyUSFM 权重，分别运行 S1 8 轮和 S3 6 轮的七张 train 图 smoke：损失从 1.9742 降至 1.9348／1.9476，检查点读写与冻结参数校验通过，S3 已经过第 6 轮微调切换。结果在 `runs/expert_typing/smoke_s1_20261004/` 与 `smoke_s3_20261004/`；这些数值仅说明训练链路可工作。S2 通过真实 ResNet18 架构冻结／BatchNorm 测试，尚未下载 ImageNet 权重进行真实数据训练。

本轮全部 149 项项目测试通过（含 10 项新增专家测试）；Ruff 检查与格式检查通过。

专项测试命令：

```bash
python -m unittest discover -s tests -p test_typing_expert.py -v
```

没有专家依赖的环境会跳过专项测试，因此应在装好 `requirements-expert.txt` 后运行。本次未启动 9 次正式训练，未读取测试图片进行预测，未产生 LLM API 费用。

当前数据已重划分为 392 张；旧训练检查点的数据或代码哈希不再匹配，须重新训练。旧运行保留原始事实，不与本次成绩混合。

新增训练调整参数和完整对照批次见 [专家训练调整指南](typing-classifier-adjustments.md)。默认训练配置不变；自定义配置使用候选清单选型。
