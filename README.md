# 肝脏超声包虫病识别

输入带诊断标签的肝脏超声图像，用视觉大模型输出患病概率 `p_he` 与简短依据。支持四种输入模式和可选阴性复核；图像保持原始字节，待测图真值只在本地评分。

**当前成果：** 已整理 219 张图片；完成 GLM 四模式、参考图数量和阈值实验，以及 GLM／DeepSeek 五种匹配配置的真实对照。OpenAI 接口已实现，尚无真实对照成绩。研究结果见[项目进展](docs/project-progress.md)，对外汇报见 [report/](report/)。

## 开始运行

在项目根目录、已激活的 Python 3.10+ 环境中：

```sh
python -m pip install -r requirements-agent.txt
python src/he_agent.py --provider glm --mode baseline \
  --split configs/reference_split_balanced.json --seed 42 --limit 3 --dry-run
```

`--dry-run` 只校验本地数据，不读取密钥或调用 API。真实运行前按[配置说明](configs/README.md)填写本地密钥，然后去掉 `--dry-run`。`--limit 0` 运行划分中的全部 211 张。

```sh
# 8 张参考图；不复核。阈值为开发集候选，不代表临床诊断阈值。
python src/he_agent.py --provider glm --mode reference \
  --split configs/reference_split_balanced.json --reference-count 8 \
  --seed 42 --limit 0 --threshold 0.25
```

换成 `--provider deepseek` 即使用该适配器默认模型。四模式、复核与离线阈值扫描的完整用法只维护在[实验指南](docs/experiment-guide.md)。

## 阅读导航

| 要做什么 | 入口 |
| --- | --- |
| 看已完成的尝试和结果 | [项目进展](docs/project-progress.md) |
| 运行、复核、调参考数、读结果 | [实验指南](docs/experiment-guide.md) |
| 理解代码与函数调用 | [架构导读](docs/agent-architecture.md) → 源码内中文注释 |
| 追溯医学提示词依据 | [医学阅读记录](docs/medical-knowledge-review.md) |
| 规划下一阶段研究 | [五项研究提案](docs/next-proposals.md) |

## 文件各归其位

```text
src/                   推理入口、概率评价、固定参考选择、模型适配器
scripts/
  data/                离线数据整理与划分生成
  analysis/            参考方案比较、冻结初判和阈值扫描
configs/               固定划分、共同队列与密钥模板
prompts/               当前提示词、来源记录及必要历史快照
tests/                 主项目离线测试
docs/                  五份核心文档
report/                对外阶段报告
experiments/           独立的专家模型与历史 Harness 原型
outputs/analysis/      核心历史结果的机器可读证据
```

本地研究资产另存于 `data/`（图片与标签）、`literature/`（文献）、`runs/<时间>/`（原始运行），不会因文档整理而删掉。`outputs/threshold_sweeps/` 存放离线重评分。密钥、原始数据、权重和生成结果不提交 Git。

## 更换数据来源或资源目录

统一修改 [configs/paths.json](configs/paths.json)，例如其中 `dataset` 指向新的统一数据集目录。相对路径按项目根目录解析，绝对路径也可使用；显式命令行参数优先。配置字段、独立本地配置和数据格式要求见 [配置说明](configs/README.md#资源路径pathsjson)。

## 维护检查

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -q
ruff check .
ruff format --check .
```

已有统一数据时不必重跑数据脚本。历史原型使用独立依赖，运行主 Agent 不需要 TinyUSFM、PyTorch 或专家权重。
