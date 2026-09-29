# 医学知识依据与阅读记录

更新：2026-09-19，现用医学prompt为v4精简版。下文保留历史阅读记录，当前四段规则映射见 `prompts/he_medical.sources.json`；[版本与实测结果](project-progress.md)。


11 sources: 7 relevant full-text readings, 3 abstracts, and 1 indexed-section reading. This is a targeted review, not a systematic review.

| ID | Source | Reading scope and adopted points |
| --- | --- | --- |
| S01 | [WHO guidelines for the treatment of patients with cystic echinococcosis: Introduction and staging table](https://www.ncbi.nlm.nih.gov/books/NBK616291/) (2025) | full relevant web sections; Introduction; Staging of cystic echinococcosis cysts. CE1–CE5 形态和 CL 定性限制；不把活动性当二分类标签。 |
| S02 | [International classification of ultrasound images in cystic echinococcosis for application in clinical and field epidemiological settings](https://pubmed.ncbi.nlm.nih.gov/12606104/) (2003) | abstract only; PubMed abstract. CL 不确定病因，不能直接当作 CE；本条仅使用摘要支持的范围。 |
| S03 | [Expert consensus for the diagnosis and treatment of cystic and alveolar echinococcosis in humans](https://pubmed.ncbi.nlm.nih.gov/19931502/) (2010) | abstract only; PubMed abstract. CE 和 AE 必须分开理解；不用治疗建议填充图像分类 prompt。 |
| S04 | [Ultrasound and Cystic Echinococcosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6199172/) (2018) | indexed full-text section excerpts only; WHO panels in detail; Differential diagnosis of parasitic liver lesions. 早期单囊病灶与单纯囊肿有重叠；下载 PDF 返回 HTML，未声称已读该下载全文。 |
| S05 | [Proposal of an ultrasonographic classification for hepatic alveolar echinococcosis: EMUC-US](https://www.wjgnet.com/1007-9327/full/v21/i43/12392.htm) (2015) | local PDF and publisher relevant sections; Diagnosis and classification; morphology definitions; Discussion; local PDF pp. 3–8. AE 存在不同形态，非典型形态与其他肿块难分；描述不是可靠单征象判定式。 |
| S06 | [Pictorial review of hepatic echinococcosis: Ultrasound imaging and differential diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11514533/) (2024) | local PDF relevant sections; Conventional US; Differential diagnosis; Table 4; local PDF pp. 4–10. 只提炼 B 模式形态及鉴别限制，不带入 CEUS 动态、病例图注或病例标签。 |
| S07 | [Cystic liver lesions: a pictorial review](https://link.springer.com/article/10.1186/s13244-022-01242-3) (2022) | full relevant web sections; Simple hepatic cyst; Complex cyst; Imaging. 薄壁单囊等组合支持普通囊肿，复杂囊肿有多种病因；只使用超声相关信息。 |
| S08 | [No cyst, no echinococcosis: a scoping review update on diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10487401/) (2023) | downloaded full PDF relevant sections; Imaging; Diagnosis; Discussion. 钙化单独不足以确诊 CE；病灶证据优先，不能将 CE 的标题口号错误用于 AE 或退变实性 CE。 |
| S09 | [Expert Reliability for the WHO Standardized Ultrasound Classification of Cystic Echinococcosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5361546/) (2017) | downloaded full PDF relevant sections; Methods; Discussion; Conclusions; printed pp. 686–691. 二维图像不能充分显示囊内结构；专家也有判读分歧。形态分期一致性不等于本 Agent 二分类准确率。 |
| S10 | [Ultrasound diagnosis of cystic echinococcosis: updates and implications for clinical management](https://pubmed.ncbi.nlm.nih.gov/38628145/) (2024) | abstract only; PubMed abstract. 影像是诊断起点；检查不到的其他临床证据不能由模型自行生成。 |
| S11 | [Early diagnosis, treatment and follow-up of cystic echinococcosis in remote rural areas in Patagonia: impact of ultrasound training of non-specialists](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0001444) (2012) | full relevant web sections; Training; Methods; Discussion. 借鉴形态识别、鉴别和专家监督。原现场项目将囊性肿块先列疑似并复核；不移植为自动阳性规则，也不把医生培训成效当 prompt 效果证据。 |

## 历史v2/v3规则来源（K01–K08）

- K01: S09, S11. 质量和可观察性限制；具体拒答/多面板处理为本项目操作规则。
- K02: S01, S04, S08, S09. CE 形态；对 CE4/5 使用支持性与不确定措辞，不宣称单张图像必然确诊。
- K03: S01, S02, S04. CL 与早期 CE 定性边界。
- K04: S03, S05, S06. AE 独立路径；EMUC 是形态模式而非 CE 分期。
- K05: S06, S07. 鉴别诊断；核对正常解剖结构身份是项目读图检查步骤，不是新增诊断算法。
- K06: S01, S06, S09, S10. 基于数据模态限制的项目推论：不补造动态、多普勒、造影或其他检查信息。
- K07: S08, S09, S11. 当前 v2 的 p_he（0–1 或 null）是研究输出规则，概率映射与辅助分类阈值不是临床指南提供或验证的。旧 v1 使用 0/1/null。
- K08: Project engineering rule. 本项目证据核对与 JSON 输出规则，不声称医学证据已验证模型效果。

## 解释与边界

- source_disagreement: WHO 2025 lists some distinctive CE4/5 appearances, whereas 2017 expert reliability work stresses etiologic uncertainty. Preserve appearance descriptions but do not equate calcification/solid texture alone with diagnosis.
- case_leakage: Only general rules are included, not individual-case captions, labels, UUIDs or source prevalence. Public-image reuse and pretraining overlap cannot be ruled out.
- physician_experience: Published expert reading studies and training studies; no interviews or clinical endorsement claimed.
- reading_scope: 7 full-text relevant-section readings, 3 abstract-only readings and 1 indexed-section reading; not a systematic review.
- excluded_clinical_workflow: FASE suspected-case referral rule is not a final binary diagnosis; omitted from model classification rules.

The Chinese prompt paraphrases general findings and adds project-specific output rules. It is not an official translation, clinician-approved tool, or validated diagnostic algorithm. No individual case answers or patient demographics were copied into the prompt.

## 概率输出更新

医学规则现用于 he_medical.txt。病因重叠但影像仍可判读时，用连续概率与 evidence 表达不确定性；无法形成有依据的估计时返回 null。沿用上述 11 份资料，不将分期、专家一致性或治疗建议转换成数值概率。尚未实施概率校准，文献依据不能替代本数据上的概率验证。

## 历史措辞修订：medical-v3-soft-guidance-probability

基于 runs/ 中本轮修订前最后四组 50 张 GLM 实验，提出“反复强调确诊门槛可能压低患病概率”的开发假设。本轮保留 K01–K08 和原有 11 份医学来源，将医学描述改为可按图像选择的辅助知识：不要求不同类型/阶段的征象同时出现，区分未显示与不存在，替代诊断须有可见依据，无法确诊不自动决定概率高低。保留基础概率协议、原拒答条件和防止编造信息的约束。此次不做知识内容的大幅精简。

v3修订当时由 knowledge 和 reference_knowledge 自动加载；baseline、reference 和模型参数保持不变。来源清单已更新版本、前版指纹和旧提示词快照路径，每次运行仍保存实际提示词和哈希。旧 runs 不修改。该修订当时只完成离线验证；后续v3/v4实测结论统一见[项目进展](project-progress.md)，不将当时状态当作当前状态；已用于分析的 50 张应视作开发观察，后续确认性评价应使用未用于修订的数据。

## 当前精简版：medical-v4-concise-probability

2026-09-19将v3的2850字符压缩为1013字符（含换行，约减少64.5%，不是token数），保留基础概率协议。当前K01–K04分别对应CE形态与活动性边界、AE形态、病因重叠、静态可观察性；具体来源映射以 `prompts/he_medical.sources.json` 为准，与上面的历史K编号不能混用。

本次精简重新核对S01的WHO形态分期/CL部分及S05的EMUC-US形态部分；保留11项历史文献。删除长分期清单、重复检查和概率方向暗示，不加入固定特征分数、患病率或诊断门槛。医学v3全文及来源清单在 `prompts/archive/`，真实运行解释以各run快照为准。

复核补充文本现为151字符的中性说明：独立观察整体与局部、核对支持/反对/未知依据，不预设概率方向，无法估计则null。修改前副本是当时用户已自行删减的版本；追溯更早实验应读原run的 `review_prompt.txt`。精简医学和中性复核均已有GLM实测，尚未显示医学增强的稳定优势；不能将文献充分性等同于prompt有效性。
