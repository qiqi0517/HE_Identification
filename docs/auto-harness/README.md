# 分型 Auto Harness 运行指南

当前入口 `scripts/experiments/auto_harness.py` 已替换旧概率版，只支持 `NO_HE / CE1 / CE2 / CE3 / CE4 / CE5 / AE`，禁止 CL。方案见 [v1-execution-plan.md](v1-execution-plan.md)。所有命令在项目根目录、`he` 环境执行；不需要 PyTorch 或专家模型权重。

```bash
conda activate he
cd /Users/qiqi/Research/HE_Identification
python -m pip install -r requirements-agent.txt
```

先确认 `configs/auto_harness.example.json` 中视觉模型、优化模型和预算。默认视觉模型为 `glm-5.3-flash`，沿用项目现有密钥读取方式。下面 `--live` 命令会调用付费 API；初始化、静态搜索、选型评分及报告均在本地运行。

## 1. 初始化新实验

```bash
python scripts/experiments/auto_harness.py init \
  --split configs/auto-harness-typing/split_v1.json \
  --dataset data/unified_typing_v2 \
  --config configs/auto_harness.example.json \
  --output runs/auto_harness/typing_v1
```

本地校验 392 张图和新划分，冻结模型配置、提示包 V0、标签／图片／参考及代码哈希。输出目录必须不存在。**不要使用旧 `runs/auto_harness/v1`；旧概率缓存、旧配置及 `--method memory` 等接口均不兼容。** 本次替换代码不会删除旧运行记录。

## 2. 采集 V0 搜索动作

先去掉 `--live` 检查计划调用量，再执行：

```bash
python scripts/experiments/auto_harness.py collect \
  --workspace runs/auto_harness/typing_v1 --role search --live
```

190 张搜索图各采集三轮：一次 BASELINE＋四种独立复核，最多 2850 次逻辑调用。初判无效时跳过复核，仍留在评分分母。参考复核使用冻结七图；医学知识复核使用完整知识；参考＋检查表在一个请求内完成。

## 3. 本地优化阈值和动作

```bash
python scripts/experiments/auto_harness.py search \
  --workspace runs/auto_harness/typing_v1
```

按初判预测类别分组，每类比较 185 条规则，生成 `P_joint`；另计算不分段的 `P_type` 供诊断。以正确数为主、模型调用数为并列条件，不调用 LLM。输出在 `search/V0/optimized.json` 和两个 `*_rules.json` 中。

## 4. 可选：LLM 累积经验优化提示词

```bash
python scripts/experiments/auto_harness.py optimize \
  --workspace runs/auto_harness/typing_v1 --live
```

最多四轮，连续两轮未改进停止。每轮带入当前最佳提示与规则、全部历史摘要、混淆及搜索错例；LLM 提案后补采受影响结果，再本地重求阈值。可能新增大量视觉调用，不能把它理解为只发四个文本请求。去掉 `--live` 只看剩余计划。

当前采用**受约束的文本块优化**：LLM 选择／重排已经审核的观察指令和检查表，不接收任意自由文本替换。检查表医学条目必须全部保留，知识正文、七类输出契约和动作输入开关只读。这避免无法用程序可靠判断的医学事实改写进入自动循环。实际渲染提示保存在 `prompt_versions/V*.json`；原始优化器请求／响应在 `requests/`，每轮上下文在 `optimizer/contexts/`，逐轮结论在 `optimizer_history.jsonl`，经验摘要与当前最佳在 `experience_summary.json`、`incumbent.json`。

可以跳过这一步先完成纯静态实验；一旦下一步冻结短名单，就不能再在这个实验目录开启优化。

## 5. 冻结候选并选型

```bash
python scripts/experiments/auto_harness.py shortlist \
  --workspace runs/auto_harness/typing_v1
python scripts/experiments/auto_harness.py collect \
  --workspace runs/auto_harness/typing_v1 --role selection --live
python scripts/experiments/auto_harness.py select \
  --workspace runs/auto_harness/typing_v1
```

最多三个候选：V0 不复核的 S0、V0 静态最优、动态优化后的最优；相同候选去重并保留别名。76 张选型图各执行三轮真实路由，每图每候选最多两次调用，**不展开所有复核动作**。冻结胜者到 `selected.json`；选型结果不反馈给优化器。

## 6. 最终测试

```bash
python scripts/experiments/auto_harness.py collect \
  --workspace runs/auto_harness/typing_v1 --role test --live
python scripts/experiments/auto_harness.py test \
  --workspace runs/auto_harness/typing_v1
```

在 119 张测试图上三轮评价冻结候选。`test_report.json` 保存准确率、各类召回、混淆矩阵、来源指标、纠错／引错、调用／token／延迟和按关联组配对 bootstrap。测试结果不改变已选胜者。产物采用 `sha256`＋`payload` 包装，指标位于 `payload` 内。

## 7. 导出并识别新图片

```bash
python scripts/experiments/auto_harness.py export \
  --workspace runs/auto_harness/typing_v1 \
  --output runs/auto_harness/typing_v1/deployment.json
python scripts/experiments/auto_harness.py predict \
  --deployment runs/auto_harness/typing_v1/deployment.json \
  --image /绝对路径/待判断图片.jpg \
  --output runs/auto_harness/typing_v1/prediction.json --live
```

只执行冻结初判→路由→接受或一次复核。部署文件仅携带七参考标签，不包含搜索／选型／测试真值；本机仍需保留参考图路径和对应版本代码。去掉 `--live` 为本地检查，不创建预测结果文件。输出路径不能已存在。

## 状态、预算与恢复

```bash
python scripts/experiments/auto_harness.py status \
  --workspace runs/auto_harness/typing_v1
```

采集或优化中断后，重新运行原命令即可复用已落盘结果。发送前先记录请求；发送后尚未落盘就中断的请求记为 `interrupted_request_billing_unknown`，不自动重发或剔除。致命 API 错误会停止采集，修复服务／凭据后可以继续，已记录失败仍计入评分。若进程被强制杀死留下 `.writer.lock`，确认文件记录的 PID 已退出后再移除锁。

默认预算为视觉逻辑调用 17175、优化提案 4；视觉已报告 token 停止值 100000000、优化器 200000。它们是允许上限而非预计必然消耗量。协议或传输重试不会超过每个请求一次；当前协议解析错误直接记错，不额外重试。token 停止依据服务方已返回 usage，最后一次调用可能跨过上限，缺失 usage 和未报告的重试无法计入；**没有金额硬预算或价格自动换算**。

遇到 token 停止时，可显式提高预算再运行原命令，例如：

```bash
python scripts/experiments/auto_harness.py budget \
  --workspace runs/auto_harness/typing_v1 \
  --vision-reported-tokens 150000000
```

也支持 `--vision-calls`、`--optimizer-reported-tokens`；修改记录写入 `budget.json`，只能提高，不能突破四轮提案上限。预算调整不改变数据／提示／模型或清除缓存。启动新优化轮前会保守检查是否有完整版本的调用额度；中途 token 停止留下的部分版本，必须恢复完成后才能参与选优或冻结。

## 免费离线演示

```bash
python scripts/experiments/auto_harness.py demo \
  --output runs/auto_harness/typing_demo
```

使用合成图片和模拟模型运行采集→静态优化→提示优化→选型→测试，不读取密钥、不调用 API。合成成绩仅验证代码，不能作为医学效果；演示部署导出被禁止。

## 本次实现验证

- 全仓 160 项离线测试通过，Ruff 与格式检查通过。
- 合成数据完整跑通静态／动态优化、选型与测试；另用模拟模型验证导出后的单图预测、每图两调用上限及部署不携带开发／测试标签。
- 当前真实 392 张数据的初始化通过；搜索 dry-run 确认 190 张×3 轮×5 次＝2850 次上限，未读取凭据或调用 API。
- 尚无本版本真实模型实验成绩。真实效果、成本及动态提示优化收益需运行后评估。
