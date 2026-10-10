# 分型 Agent：检查表复核提示与运行方式

更新：2026-10-04。检查表提示及独立复核参考／知识开关已实现，直接更新当前文件，不另建版本。仍按初判 confidence 触发最多一次独立复核。默认复核知识现为 checklist，默认参考为 same；阈值默认仍是 0.8，下面实验显式使用待验证的 0.6。

## 建议优先比较两种不同信息组合

旧搜索集实验中，参考＋知识初判后，以同样参考＋知识复核几乎覆盖全部图片，净损失 4 张正确。这提示重复相近输入未必有收益，但旧结果不代表当前新数据与七类协议的效果，必须重跑评价。

| 配置 | 初判输入 | 复核输入 | 要检验的问题 |
| --- | --- | --- | --- |
| R0 对照 | 当前选定输入 | 通用中性提示、同参考／知识开关，不附检查表 | 保留现有流程作为对照 |
| R1 首选候选 | 原图＋固定 7 张参考，可加知识 | **原图＋简短鉴别检查表，不加参考图** | 减少单张参考外观对二次判断的重复影响 |
| R2 候选 | 当前选定输入 | 原图＋固定 7 张参考＋同一检查表 | 检验结构化对照是否比当前泛化复核提示有效 |

R1/R2 均不输入初判类别、confidence、理由或“上一轮可能错了”的暗示；保留独立复核性质。初判信息仅供外部路由。复核仍输出自身类别与 confidence，最多一次并替换初判；不因想避免改变结果而复制初判。

## 建议复核提示词

下面为提示概要。实际文件为 `prompts/he_typing_review.txt`（中性复核指令）和 `prompts/he_typing_review_checklist.txt`（检查表）；程序在基础七类输出协议后拼接两者：

> 请独立检查最后一张待判超声图。仅允许 NO_HE、CE1、CE2、CE3、CE4、CE5、AE；禁止输出 CL。先确认图片可判读范围，再核对可见结构：单囊及囊壁／囊内回声，多发子囊或分隔，脱落内膜，杂乱实变样内容物，钙化及后方声影。结合可见形态鉴别；不以单一钙化或实性外观直接判断 AE，不把缺少典型征象直接判断为 NO_HE。若有参考图，只比较可见结构，不以设备风格、文字、亮度或整体相似度替代判断。选择最有依据的七类结论；若确实无法判断，返回 abstain。不要预设必须改变判断或提高 confidence。只输出指定 JSON，findings 包含 1–5 条简短可见依据及关键不可确认点，不输出完整推理过程。

知识条目来自项目现有 PPT 转录与知识 prompt；`prompts/he_typing_review.sources.json` 已冻结基础协议、中性指令、检查表的哈希及医学来源。保持 `status/predicted_type/confidence/findings` 输出契约，不把 NO_HE 当成“不确定”类别。输出格式错误与分类错误分开统计，但均保留在总准确率分母中。

## 参考图怎么改

第一批仍使用**新划分的固定 7 张参考，每类一张，顺序一致**。R1 整体关闭参考，R2 使用同一七张；这样可以明确区分“检查表”和“是否加参考”的贡献。人工核对参考图可判读性及图内文字，如果需要换图，只在开发侧操作并再次冻结划分，不根据测试正确率挑参考。

暂不只挑“初判同类”的参考：这容易强化初判偏向。未来如研究 top-2 类参考检索，需要独立定义候选生成和固定参考池规则；不能按真实类别挑图，也不能从待判集合动态抽取带真值图片作为参考。单纯增加参考到 14 张需要重新分配参考集合，不直接拿搜索／选型／测试图补齐。

## 实现接口与评价方式

已实现独立于初判的 `--review-reference none|same` 与 `--review-knowledge none|same|checklist`。`same` 只继承初判实际使用的参考／知识；初判不使用参考时，复核 same 也不添加参考。`checklist` 使用基础输出协议＋中性指令＋检查表，不继承初判完整知识。`none` 不附加知识或检查表。

`--review-prompt` 可替换中性复核指令，所选知识仍按开关追加。日志保存最终复核 prompt、参考图片 ID／顺序／哈希和知识来源。`review_prompt.txt` 是实际发送的完整复核文本，`review_references.json` 是实际复核参考清单。

只在新搜索集比较 R0/R1/R2，使用相同初判结果、相同触发图片和相同模型参数，避免把初判随机变化当成复核收益。统一阈值可先固定 0.6 做输入对照；这是开发候选，旧数据的 0.6 收益不能迁移为新数据结论。必须保留不复核基线。

评价包括总体准确率、复核纠错数／引错数／净正确数、按初判类别的净收益、调用量与 token；CL 已非法，不再设置 CL 路由。候选冻结后用新选型集选定，再评价测试集。重新划分可能将旧搜索图片移入新测试，当前结果仍属内部探索，不宣称全新未接触过的临床测试。


## 直接运行

在项目根目录和已配置 GLM 密钥的 `he` 环境运行。以下命令会调用 API；本次开发只运行了 dry-run 和模拟请求测试。

### R1：检查表复核，不加参考图（建议先运行）

```bash
conda activate he
cd /Users/qiqi/Research/HE_Identification
python src/he_agent.py \
  --provider glm --model glm-5.3-flash \
  --mode reference_knowledge --reference-count 7 \
  --review --review-threshold 0.6 \
  --review-reference none --review-knowledge checklist \
  --role search --limit 0 --seed 42 \
  --output runs/typing_review_checklist/r1_no_reference
```

初判：原图＋7 张参考＋完整分型知识。仅在初判有效且 confidence < 0.6 时复核，复核：原图＋基础输出协议＋中性指令＋检查表；不输入初判结果，不输入参考图或完整知识。遍历新搜索集全部 190 张图片。0.6 是开发候选，并非已验证最佳阈值。

### R2：检查表复核，保留相同七张参考

```bash
python src/he_agent.py \
  --provider glm --model glm-5.3-flash \
  --mode reference_knowledge --reference-count 7 \
  --review --review-threshold 0.6 \
  --review-reference same --review-knowledge checklist \
  --role search --limit 0 --seed 42 \
  --output runs/typing_review_checklist/r2_same_reference
```

初判设置与 R1 相同；复核另提供初判相同的七张参考图及顺序。

### 不复核对照

```bash
python src/he_agent.py \
  --provider glm --model glm-5.3-flash \
  --mode reference_knowledge --reference-count 7 --no-review \
  --role search --limit 0 --seed 42 \
  --output runs/typing_review_checklist/no_review
```

若需要 R0 通用中性对照，在 R2 命令中将 `--review-knowledge checklist` 改成 `--review-knowledge same`，并换输出目录。它使用当前通用指令，不是重放历史旧提示。

各命令添加 `--dry-run` 并换新输出目录即可仅做本地校验、保存最终提示和输入清单，不调用 API。`--limit 3` 可以先小批量真实检查；输出目录不能已存在。

**比较边界：** 当前基础 Agent 不支持复用另一个运行的初判缓存。分开运行 R1/R2 会分别产生初判，远程模型存在随机性，相同 seed 只固定样本顺序。先比较每个运行内部的初判→复核净收益，并补充重复实验；如需严格相同初判／触发图片的输入消融，需另实现缓存回放，不把这里的两个独立运行描述为严格配对因果实验。

实现验证：153 项项目测试通过；新搜索集 190 张的 R1/R2 dry-run 均通过，核验实际复核参考分别为 0／7 张，复核文本包含检查表且不继承完整知识。未发起付费调用。
