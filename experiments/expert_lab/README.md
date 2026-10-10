# 专家模型独立实验区

2026-10-04 更新：基于此处已校验的 TinyUSFM 编码器，七分类专家训练已在主项目 `src/typing_expert/` 实现，见 [运行指南](../../docs/expert/typing-classifier-v1-runbook.md)。下文描述的是早期特征检索实验。

2026-09-18。这里是新建的实验子项目，未接入或改动现有 Agent。

**已实际跑通 TinyUSFM 预训练超声编码器，并实现可调用的相似参考图工具。当前没有得到经过包虫病训练和验证的疾病概率专家。** `p_he: null` 是有意保留的能力边界，不是解析失败。

先读 [调查与实验结论](research-report.md)，再看 [真实运行结果](results/tinyusfm_reference_smoke/summary.json) 和 [工具输出示例](results/tinyusfm_reference_smoke/tool_examples.json)。

## 已实现的内容

| 文件 | 用途 |
|---|---|
| `src/ultrasound_encoder.py` | 严格校验公开权重 SHA-256、构造编码器、只读图像预处理、192 维特征和余弦检索 |
| `src/run_experiment.py` | 校验固定划分，仅在指定划分的参考池上运行实验（当前 v2 为 8 张），保存特征、统计和工具输出 |
| `src/expert_tool.py` | 独立工具入口；输入图片，输出相似参考图及参考标签，排除字节相同的图片 |
| `src/he_checkpoint.py` | HE_DCNN 五分类 checkpoint 兼容性探针、类别概率求和及未就绪状态；不是已完成的 HE_DCNN 推理复现 |
| `tests/test_expert.py` | 权重一致性、概率边界、数据隔离、缺权重拒答等验证 |
| `weights/TinyUSFM.pth` | 已下载的真实公开超声预训练权重，约 22 MB，不纳入 Git |
| `research/` | 官方仓库元信息及作者源码快照，保留检查依据 |
| `results/` | 实际运行记录、测试日志、原文件保护核验 |

## 运行

本机已配置并验证本目录的 `.venv`，可直接执行下面的运行命令。其他机器推荐 Python 3.12，首次安装：

```bash
cd /Users/qiqi/Research/HE_Identification/experiments/expert_lab
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

权重已经保存在本机 `weights/`；若从 Git 获取本实验，需要另外下载：

```bash
.venv/bin/gdown 1P_wDuU-StALLJEBaymioaYu52CScW-l- -O weights/TinyUSFM.pth
```

重新运行参考池实验（使用新输出目录，防止覆盖记录）：

```bash
.venv/bin/python -B src/run_experiment.py \
  --project /Users/qiqi/Research/HE_Identification \
  --checkpoint weights/TinyUSFM.pth \
  --output results/repeat_01
```

独立调用工具，以下示例仍只用一张参考图：

```bash
.venv/bin/python -B src/expert_tool.py \
  --project /Users/qiqi/Research/HE_Identification \
  --checkpoint weights/TinyUSFM.pth \
  --image ../../data/unified/images/5e354e11-c49b-552f-bf2c-327067d1ec07.jpg
```

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

工具输出中的 `cosine_similarity` 只表示特征相似度，不是 `p_he`。当前工具默认取全池最近的 4 张，**尚未控制 2 阳性 + 2 阴性的组成，也未处理查询与参考图之间的近重复/患者关系**，因此不能直接替换主实验的固定参考图方案。

图像只在模型输入的内存中按作者方案缩放到 224×224；原文件不覆盖、不压缩。原 v1 编码实验没有对池外的 175 张测试图推理或评分。当前默认划分已变为 v2：参考池 8 张（4+4）、测试集 211 张；旧 44×192 索引仍是历史记录，不能用于新划分的检索，须运行本脚本写入新目录重建。主项目未增加模式、未修改提示词、未调用付费 API。
