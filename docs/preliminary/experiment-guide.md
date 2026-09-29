# 实验指南

命令均从项目根目录执行。主入口是 `src/he_agent.py`；数据准备只在新增／重建数据时运行，见 [scripts/data/README.md](../scripts/data/README.md)。

## 1. 环境、密钥与本地检查

使用 Python 3.10+，安装 `requirements-agent.txt`。三个 provider 为 `glm`、`deepseek`、`openai`，代码内默认模型分别为 `glm-5.3-flash`、`deepseek-flash`、`gpt-5.6-terra`；配置名称不等于账户已验证可用。GLM／DeepSeek 有真实运行，OpenAI 仅完成离线适配验证。

密钥配置只看 [configs/README.md](../configs/README.md)：非空环境变量优先，其次 `configs/api_keys.local.json`。不要写入 example、提示词或实验快照。`.env` 不会自动读取。

```sh
python src/he_agent.py --provider glm --mode baseline \
  --split configs/reference_split_balanced.json --seed 42 --limit 3 --dry-run
```

校验全部有标签图片、清单哈希与划分，再选取样本；dry-run 不读取密钥、不联网、没有预测成绩。

默认数据、提示词、密钥位置和输出目录统一由 [configs/paths.json](../configs/paths.json) 读取，具体覆盖规则见 [配置说明](../configs/README.md)。更换数据位置后，先运行上面的 dry-run；路径配置不会跳过标签／图片哈希／固定划分校验。

## 2. 四模式与固定队列

| mode | 提示词 | 参考图 |
| --- | --- | --- |
| baseline | he_binary.txt | 无 |
| reference | he_binary.txt | 有 |
| knowledge | he_medical.txt | 无 |
| reference_knowledge | he_medical.txt | 有 |

正式对比始终显式带 `--split configs/reference_split_balanced.json`。默认 `default_split=null` 时，增强模式必须通过 `--split` 提供划分；baseline 不带划分会选全部 219 张，不能直接与 211 张结果比较。`--seed` 只固定本地抽样和顺序，不固定 API 生成。

```sh
python src/he_agent.py --provider glm --mode reference \
  --split configs/reference_split_balanced.json --reference-count 8 \
  --seed 42 --limit 0 --threshold 0.25
```

这是一次付费 API 运行。换 provider 即可复用流程；改变模式时，无参考模式须去掉 `--reference-count`。`--limit` 默认 3，0 表示全部；相同数据、划分、seed、limit 对应相同图片和顺序。`--output` 必须是尚不存在的新目录，默认新建时间命名目录。

### 参考池与实际参考数

当前 balanced 池为 8 张，4 阳／4 阴，测试队列 211 张（101 阳／110 阴）。阳性 HE_DCNN／Bioengineering 各 2 张；阴性 HE 原类别 3／4 各 2 张。总体来源为 HE 6、Bio 2，不是三个来源均衡。网页来源缺患者信息，仍按来源整组留在测试侧。

默认实际发送 4 张；`--reference-count 6` 或 `8` 扩展同一池内的固定前缀，不改变测试队列，每个前缀均类别平衡。参考池不是训练集，给满 8 张时没有检索选择空间。

| 配置 | 用途 |
| --- | --- |
| reference_split_balanced.json | 当前运行优先显式使用 |
| reference_split.json / reference_split_diverse.json | 复现历史参考选择，成员与当前不同 |
| reference_split_v1.json | 早期 44 图池／175 图评价，仅供历史复现 |
| reference_*_cohort.json | 旧、新划分完整交集，供离线配对比较 |

生成器 `scripts/data/reference_balanced.py` 只按来源、类别、组隔离和种子选择，不读模型得分。推理时只调用 `src/reference_split.py` 的校验和选择函数，无需重新生成划分。未知患者与 dHash 分组不能证明患者完全独立。

## 3. 选择性复核（Agent 2.0）

```sh
python src/he_agent.py --provider glm --mode reference \
  --split configs/reference_split_balanced.json --reference-count 8 \
  --seed 42 --limit 0 --review --review-threshold 0.1 --threshold 0.3
```

先用选定模式生成初判，仅当 **0.1 < p初判 < 0.3** 才复核一次。复核仍看原图和相同参考图，追加 `he_review.txt`；不发送初判概率、理由或测试真值，避免锚定。最终采用复核输出，包括错误或 null；不取两次最大值、不静默回退。

| 参数 | 当前 CLI 默认 | 作用 |
| --- | --- | --- |
| --review | 关闭 | 开启后每图最多两次逻辑调用 |
| --review-threshold | 0.1 | 严格复核下界 |
| --threshold | 0.5 | p≥t 判阳性；同时是在线复核上界 |
| --max-tokens | 8192 | 输出预算；可能包含推理 token |
| --timeout / --retries | 180 秒／2 次 | 单次 HTTP 超时与重试上限 |

开启复核必须满足 `0 <= review_threshold < threshold <= 1`。例如分类阈值 0.2 时，可显式设下界 0.1。文档中的候选值不会自动改写程序默认值；不同模型需各自验证。

## 4. 输出、指标与异常

| 文件 | 阅读用途 |
| --- | --- |
| run.json、samples.json | 参数、输入指纹、实际样本与顺序 |
| prompt.txt、split.json、references.json | 复现当次输入；具体文件依模式生成 |
| predictions.jsonl | 每图最终概率、真值、依据、状态；复核时含两阶段结果 |
| stages.jsonl、review_prompt.txt | 开启复核时的独立阶段记录与复核提示词 |
| metrics.json | 总体／分来源评分；复核时含初判、最终及配对结果 |
| validation.json | dry-run 的本地校验结果，不是医学成绩 |

正常输出是 `{"p_he": 0.7, "evidence": "简短影像依据"}`。null 为合法弃权；截断、格式或传输问题为 error，二者均不当阴性。终端逐图显示真值与预测，最后显示平均 Log loss、Brier 和覆盖率。

- **Log loss** 按有效样本求平均，不限定在 [0,1]；仅计算对数时截断极端值以避免 log(0)。**Brier** 为概率平方误差的均值。二者越低越好。
- **AUC** 评价正负排序，单一类别时无定义；准确率、FN／FP 等依赖分类阈值。
- 总体分数同时报告覆盖率，再比较共同有效样本及来源子集。不同样本组成的 loss 不能只因“都是均值”就直接比较。
- `--threshold` 离线重评分不改变原概率；开启在线复核后重新运行则会改变门控和最终输出。
- 每图完成即落盘；中断前已保存的数据保留，但程序没有自动断点续跑。退出码 0 为完成／仅弃权，1 为出现逐图错误，2 为启动校验失败。

## 5. 冻结初判，离线扫描阈值

```sh
python scripts/analysis/sweep_thresholds.py \
  --run runs/20260920_095918_220159 \
  --review-thresholds 0.05 0.1 0.15 \
  --review-upper 0.2 --decision-thresholds 0.2 0.25 0.3
```

只读一组已完成运行的初判／复核缓存，不读密钥、不联网、不改原 run。这里复核上界和最终分类阈值可独立设置，每个分类阈值自动加入“不复核”对照。需要的复核缓存缺一条，整组候选标记 unavailable，不能回退初判来伪造完整对照。

新结果位于 `outputs/threshold_sweeps/<时间>/`：`summary.md` 快速阅读，`results.json` 完整指标，`manifest.json` 指纹与参数，`source/` 原始输入快照。不能跨 run 拼接复核缓存，也不能把缓存内的最优候选说成独立验证结果。

## 6. 比较不同参考划分

```sh
python scripts/analysis/compare_reference_runs.py \
  --old-run runs/旧运行目录 --new-run runs/新运行目录 \
  --cohort configs/reference_balanced_cohort.json \
  --output outputs/reference-comparison.json
```

填写两组真实、完整且与 cohort 对应的单次运行。脚本核对模型、模式、参考数、提示词、阈值等一致，再比较冻结的旧／新测试交集；不接收复核运行。避免将“难图移进参考池、不再评分”误认为提升。分析脚本从旧 `src/compare_reference_runs.py` 移至此目录。

## 7. 当前比较边界

现有图像已用于提示词和阈值开发；当前候选、重复实验及两模型成绩见[项目进展](project-progress.md)。密钥、预算和参数需要实测验证，程序测试与 dry-run 只证明工程契约，不证明诊断效果。
