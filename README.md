# 肝脏超声包虫病概率识别 Agent

面向本科研究项目的固定工作流：读取有诊断标签的超声原图 → 构造实验输入 → 调用视觉模型 → 校验患病概率 → 保存与评价。支持 **GLM-5.3-Flash / GPT-5.6 Terra × 四种模式**，共八组配置。

已完成 GLM 四模式真实实验，包括旧划分的完整 175 张测试图；GPT 接口已通过离线验证，真实两模型效果比较尚未完成。最新结果、提示词修订和专家模型探索见 [项目进展](docs/project-progress.md)。

## 快速开始

在 Python 3.10+ 环境运行：

```sh
cd /Users/qiqi/Research/HE_Identification
python -m pip install -r requirements-agent.txt

# 无需 API Key，检查组合版的全部固定测试图
python src/he_agent.py --provider openai --mode reference_knowledge \
  --split configs/reference_split.json --limit 0 --dry-run
```

真实调用需要在本地配置 `OPENAI_API_KEY` 或 `ZHIPU_API_KEY`；具体操作见 [实验指南](docs/experiment-guide.md)。程序不自动读取 `.env`。运行产生新目录，不覆盖已有实验。

## 四种模式

| --mode | 医学知识 | 带标签参考图 | 每次测试图 |
| --- | --- | ---: | ---: |
| baseline | 无 | 0 | 1 |
| reference | 无 | 4 | 1 |
| knowledge | 有 | 0 | 1 |
| reference_knowledge | 有 | 4 | 1 |

统一输出 `p_he`（患包虫病概率），程序计算 `p_non_he=1-p_he`。概率未经校准；无法估计为 `null`。数据真值仍为 0/1。当前划分 v2 为参考池 8 张（4 阳性＋4 阴性）、测试集 211 张。默认仍发送原来的四张示例，可用 `--reference-count 6` 或 `8` 扩展。两模型比较必须使用相同划分；旧 175 张成绩和新 211 张成绩不能直接混比。旧划分保存在 `configs/reference_split_v1.json`，变更依据与复现命令见 [参考池调整](docs/reference-pool-v2.md)。

新增来源覆盖对照 `configs/reference_split_diverse.json`：保留原四张默认示例，将额外两张重复阳性换成 HE_DCNN 原类别 0、1。请显式选择新 split，并在冻结的 209 张共同测试图上比较，不能直接比较两边各自 211 张总体分数。新版本尚无真实效果结论，命令见 [参考选择调整](docs/reference-diversity-adjustment.md)。

## 项目结构

```text
src/                  Agent、提示词加载、固定划分校验
  models/             模型契约、注册表、GLM/OpenAI 适配器
prompts/              两份现用概率提示词及医学来源清单
configs/              固定参考池/测试集划分
tests/                概率、模型协议、八组配置和数据隔离测试
scripts/data/         数据整理与划分生成脚本及说明，供复现
docs/                项目记录、使用指南、架构、医学依据、后续提案
```

本地另有 `data/`、`literature/`、`runs/`，分别存放数据、文献和运行记录，已被 Git 忽略。目录结构中的 `docs/` 与 `src/` 同级。

## 文档导航

- [项目进展](docs/project-progress.md)：已完成尝试、数据构成、验证状态和文件迁移。
- [实验指南](docs/experiment-guide.md)：密钥配置、八组命令、输出和指标。
- [Agent 架构与函数导读](docs/agent-architecture.md)：推荐先读，解释职责分工与调用过程。
- [医学知识阅读记录](docs/medical-knowledge-review.md)：11 份来源及采用范围。
- [下一步研究提案](docs/next-proposals.md)：五项可执行 proposal 及阶段进入条件。
- [数据脚本说明](scripts/data/README.md)：如何复现数据整理，而非重新运行全部历史操作。

## 开发检查

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

运行时只额外依赖 Pillow，网络调用使用 Python 标准库。历史 PDF 整理脚本另需 pypdf。数据和模型参数边界详见实验指南。
