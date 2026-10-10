# 非包虫病与包虫病分型数据集 v2

更新：2026-10-04。当前数据入口为 [data/unified_typing_v2](../../data/unified_typing_v2/README.md)，主标签为 `labels/labels_typing.txt`。替代六类草稿 `unified_typing_v1`；旧版本仅保留作追溯，不再作为当前任务输入。

## HE_DCNN 原类别核查与纠正

此前仅从旧 `ce5_label` 字段寻找分期，遗漏了 HE_DCNN 的原始类别证据。本次对照 `data/external/HE_DCNN/data/meta/{train,val,test}.txt`、`data/train/0..4`，按原图 SHA-256 与 `uuid_mapping.jsonl` 逐图核验，恢复以下信息：

| HE_DCNN 原类别 | 已整理图片数 | 原仓库含义 | 当前处理 |
| --- | ---: | --- | --- |
| 0 | 8 | AE，泡型包虫病 | 纳入 AE |
| 1 | 8 | 代码显示为 CE1，属于两组 CE 活动性分类之一 | 无可靠逐图 CE1–CE5 分期，排除 |
| 2 | 6 | 代码显示为 CE2，属于两组 CE 活动性分类之一 | 无可靠逐图 CE1–CE5 分期，排除 |
| 3 | 7 | 良性非包虫病局灶性肝病变 | 纳入 NO_HE |
| 4 | 7 | 恶性局灶性肝病变（非包虫病） | 纳入 NO_HE |

原代码 `tools/test.py` 的 `AIresult_postpress` 将 0 归入 AE、1/2 合并为 CE、3/4 归入非包虫病；五类显示名称位于该文件约第 385 行。参考[作者原仓库](https://github.com/qigtang/HE_DCNN)及[原始代码](https://github.com/qigtang/HE_DCNN/blob/main/tools/test.py)。通用 `imagenet.py` 中的鱼类等 ImageNet 名称是框架残留，不是本任务标签依据。

[原研究摘要](https://pubmed.ncbi.nlm.nih.gov/37507196/)说明其第四阶段区分活动／过渡期 CE 与非活动期 CE。因此，仓库显示的“CE1、CE2”不能直接解释为用户 PPT 的“单囊型、多子囊型”。现有原始逐图资料未给出足以恢复 WHO CE1–CE5 的细分标签；保留原编号及粗分组证据用于追溯，不按外观补标签，也不猜测两组编号顺序。

## 疾病类别与“不患病”的含义

**肝泡状棘球蚴病属于包虫病，标签是 AE。** 世界卫生组织将囊型和泡型列为人类包虫病的主要形式，见 [WHO 说明](https://www.who.int/news-room/questions-and-answers/item/echinococcosis)。它与囊型 CE1–CE5 分期不同，但不能标为阴性。

本任务的“不患病”编码为 `NO_HE`，确切含义是“非包虫病”，不表示没有其他肝病。按用户要求，全部 114 张旧二分类阴性合并到该类：HE_DCNN 14 张，Normal 来源 100 张。保留 Normal 来源标签依据及其缺少逐人包虫病专项排除证据的限制，不把来源标签描述为新临床确诊。

## 当前数据与排除情况

| 标签 | 数量 |
| --- | ---: |
| NO_HE | 114 |
| CE1 | 77 |
| CE2 | 67 |
| CE3 | 24 |
| CE4 | 48 |
| CE5 | 32 |
| AE | 30（医院 22 + HE_DCNN 8） |
| **正式入选** | **392** |

对原 219 张与医院 190 张合计 409 张输入，排除 17 张：

- 14 张 HE_DCNN CE 粗分组图片：患病已知，具体 CE 分期不明确。
- 3 张网页来源阳性图片：没有明确分期。
- 47 张医院“多子囊型可能性大”现按用户要求作为 CE2 ground truth 纳入，不再排除。原目录和 `source_status: probable` 保留，新增 `user_assigned_ground_truth` 标记；这是任务真值指定，不表示新增临床确诊。

上述 17 张不在当前 `images/` 或主标签中，仅保存排除记录及原始来源；原始文件不删除。未重新引入旧的 37 张无逐图标签图片和 8 条既有 review 排除记录。

## 文件与复现

- `images/`：仅 392 张正式入选图片，字节与输入一致，UUID 延续。
- `labels/labels_typing.txt`、`labels/manifest.jsonl`：仅 392 条七类标签和来源证据。
- `labels/excluded_records.jsonl`：17 条排除记录，不作为模型输入。
- `labels/input_audit.jsonl`：409 条输入审计记录；其中排除记录的 `image` 是候选统一名，不代表在当前 images 中存在，原图通过来源路径追溯。
- `labels/legacy_snapshot/`：旧标签和完整来源／患者证据，仅用于审计，不作为训练入口。
- `summary.json`、`checksums.json`：统计与产物哈希。

当前数据和划分已经在原路径覆盖。需要重建并覆盖当前数据时：

```bash
python scripts/data/merge_hospital_typing.py \
  --legacy data/unified \
  --hospital data/samples \
  --he-source data/external/HE_DCNN \
  --output data/unified_typing_v2 --overwrite
```

仍需在实验前审核患者关联、近重复、图内身份和诊断文字。当前内部探索划分为参考 7／搜索 190／选型 76／测试 119，专家映射为训练 197／验证 76／测试 119，见[运行指南](../agent/typing-agent.md)。一个 56 张 Normal 保守组整体进入测试，分组完整性优先于精确分层。重新划分包含历史开发图片，不宣称全新未暴露的临床测试。本次没有新的付费调用；自动优化研究流程见[执行方案](../auto-harness/v1-execution-plan.md)。

Agent 的输出仅限七类，CL 被解析器拒绝；弃权仍是独立状态。原数据、划分和文档路径保持不变，没有新增迭代版本。
