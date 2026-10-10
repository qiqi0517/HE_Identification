# 专家训练调整：运行与选型

2026-10-04。根据 S1/S2/S3 的 seed=42 验证结果新增可配置训练参数，原训练默认值不变。数据仍为 train 197／validation 76／test 119，不重新划分。当前只完成实现、测试和七张训练图的 smoke 检查，尚未运行下列正式调整实验。

## 改动与比较目的

| 候选名 | 配置 | 相对各自默认对照的变化 |
| --- | --- | --- |
| S1_default | TinyUSFM 冻结特征 | 原配置 |
| S1_standard | 同上 | `--feature-scaling standard`，按训练特征逐维标准化 |
| S1_long | 同上 | 最多 200 轮，验证损失早停，patience=20；保持原始特征 |
| S2_default | ResNet18 冻结特征 | 原配置 |
| S2_wd1e3 | 同上 | weight decay=0.001 |
| S2_wd1e2 | 同上 | weight decay=0.01 |
| S2_weighted | 同上 | 交叉熵类别权重与训练频数平方根成反比 |
| S3_default | TinyUSFM 最后两层微调 | 原配置 |
| S3_head1e3 | 同上 | 第 6 轮后分类头学习率保持 0.001，编码器仍为 0.00001 |

第一轮分别比较，不把标准化、类别加权和学习率调整全部叠加。S1_long 是一项“延长训练及更改停止规则”的组合对照，不单独归因于轮数或早停指标。候选及预期参数已固定在 `configs/expert-typing/adjustment-candidates.json`；不依据测试成绩扩充候选。

## 一次运行全部九组

已激活 he 后，从项目根目录执行：

```bash
conda activate he
cd /Users/qiqi/Research/HE_Identification
bash scripts/experiments/run_expert_adjustments.sh 42
```

这会运行九个候选各一次，共 9 次训练。完成后补齐另外两个种子：

```bash
bash scripts/experiments/run_expert_adjustments.sh 43 44
```

也可以在尚未开始时一次提交 `42 43 44`，共 27 次训练。脚本遇到失败立即停止。它只训练并验证，不调用 LLM，也不读取测试图做预测。输出目录为 `runs/expert_typing/adjustments/<候选名>/<种子>/`，不能已经存在；失败后根据下方单组命令继续尚未完成的配置，不盲目从头覆盖。

原三个模型结果保留在 `runs/expert_typing/v1/`。新代码改变了实现指纹，因此本轮在独立实验目录重跑三个默认对照，所有参与选型的运行使用一致实现。旧权重的严格代码指纹检查仍保留，当前加载器不会绕过检查载入旧实现权重；历史日志和分析不受影响。

## 单独运行各项调整

下列均为 seed=42，已由批量脚本完成的组合无需重复执行。

```bash
# S1：只增加训练特征标准化
python scripts/experiments/train_typing_expert.py train \
  --variant S1 --seed 42 --feature-scaling standard \
  --output runs/expert_typing/adjustments/S1_standard/42

# S1：原始特征，延长训练，按验证损失早停
python scripts/experiments/train_typing_expert.py train \
  --variant S1 --seed 42 --epochs 200 --early-stop-metric loss --patience 20 \
  --output runs/expert_typing/adjustments/S1_long/42

# S2：两档更强正则化
python scripts/experiments/train_typing_expert.py train \
  --variant S2 --seed 42 --weight-decay 0.001 \
  --output runs/expert_typing/adjustments/S2_wd1e3/42
python scripts/experiments/train_typing_expert.py train \
  --variant S2 --seed 42 --weight-decay 0.01 \
  --output runs/expert_typing/adjustments/S2_wd1e2/42

# S2：只增加类别加权，不叠加重采样
python scripts/experiments/train_typing_expert.py train \
  --variant S2 --seed 42 --class-weight inverse_sqrt \
  --output runs/expert_typing/adjustments/S2_weighted/42

# S3：仅提高微调阶段的分类头学习率
python scripts/experiments/train_typing_expert.py train \
  --variant S3 --seed 42 --finetune-head-lr 0.001 \
  --output runs/expert_typing/adjustments/S3_head1e3/42
```

默认对照命令仍为 `train --variant S1 --seed 42 --output runs/expert_typing/adjustments/S1_default/42`，S2/S3 替换相应模型名即可。完整入口前缀均为 `python scripts/experiments/train_typing_expert.py`。

## 选型与冻结测试

全部九候选×三种子完成后，用声明的候选清单选型：

```bash
python scripts/experiments/train_typing_expert.py select \
  --runs runs/expert_typing/adjustments \
  --candidates configs/expert-typing/adjustment-candidates.json \
  --output runs/expert_typing/adjustments/selection.json
```

检查每组实际参数是否符合声明、三个种子是否使用相同超参数及预训练权重、全部运行的数据和代码是否一致。按三个种子的验证正确数总和排名，并列看平均宏召回率，再按清单顺序；部署权重固定 seed=42，不挑最好的种子。只有 seed=42 时不能执行最终选型。

```bash
python scripts/experiments/train_typing_expert.py evaluate \
  --freeze runs/expert_typing/adjustments/selection.json \
  --output runs/expert_typing/adjustments/test_selected
```

需要预设候选的测试对照时，可指定 `--candidate S1_standard --seed 43` 并使用新的输出目录。同一个 S1/S2/S3 下有多个候选时，不要仅用 `--variant`，否则会因含义不明确而拒绝执行。

## 实现约定

- 标准化仅支持冻结编码器 S1/S2，计算 train 特征的逐维均值和总体标准差；标准差小于 1e-6 时用 1，避免放大近零噪声。validation/test 不参与拟合。统计量写入 `feature_scaling.json` 及检查点 buffer，缓存、完整图像预测和重新加载后使用同一变换。S3 的特征在训练中变化，本次不对其开启标准化。
- 类别权重只由训练真值频数计算，归一化至七类平均权重为 1。训练损失按权重分母汇总，验证损失仍为普通未加权交叉熵，保证跨配置可解释。
- 早停与检查点选择分离：`--early-stop-metric loss` 只控制何时结束训练；所有配置的最佳检查点仍按“验证正确数最多→普通验证交叉熵最小→更早 epoch”确定。因此最后一轮未必是保存的检查点。
- `history.jsonl` 新增实际优化器学习率和未改善轮数；所有新参数写入 `config.json`，非默认参数标记为自定义实验。旧 `select` 默认仍只接受原 S1/S2/S3 协议；`--candidates` 才能选择声明的调整配置，smoke 一律拒绝。
- 学习率／weight decay 拒绝非有限数值和非法范围。候选清单不修改数据划分，不读取测试成绩参与选择。
