# 数据处理脚本

此目录保存数据清洗、标签整理、合并以及实验划分准备脚本；Agent 入口为 src/he_agent.py。命令从项目根目录运行。

| 脚本 | 用途 |
| --- | --- |
| merge_hospital_typing.py | 合并旧 UUID 数据和医院分型样例，保留明确／疑似标签与溯源，输出独立版本 |
| merge_he_dcnn.py | 整理 HE_DCNN 图片、去重并生成二分类标签 |
| prepare_bioengineering_us.py | 校验 Bioengineering PDF 提取图片、筛选肝超声并整理标签 |
| prepare_web_us_filtered.py | 历史一次性操作：排除指定箭头来源并整理网络图片；当前数据已经处理，不应重复执行 |
| merge_processed_uuid.py | 将三个处理后来源合并，以 UUID 命名，并单独保存未标注图片 |
| merge_normal_labels.py | 将正常来源标签 0 纳入合并数据集的主标签表 |
| prepare_reference_split.py | 按患者/来源/重复组建立平衡参考池，生成固定划分；从 src/reference_split.py 拆出 |
| reference_balanced.py | 按当前来源／类别配额建立八图参考池及旧新测试交集 |
| reference_diversity.py | 基于原划分生成来源覆盖对照和共同测试队列；从 src/ 移入 |

使用 `python scripts/data/脚本名.py --help` 查看参数。多数脚本拒绝覆盖已有输出。数据已整理好时无需再次运行这些脚本；运行 Agent 直接读取 data/unified。

划分配置已经存在时，Agent 只读取 configs/ 中的 JSON，调用 src/reference_split.py 的 validate_split 和 select_references，无需执行这两个生成脚本。生成器供重建、扩充数据或准备新实验使用，不是废弃文件。

查看参数（不生成文件、不调用 API）：

```sh
python scripts/data/prepare_reference_split.py --help
python scripts/data/reference_diversity.py --help
```

依赖：多数脚本只需 Pillow；PDF 校验脚本另需 `pypdf`。历史一次性脚本按原始批次编写，输入清单或目录已迁移时，应先阅读文件顶部说明与 main 的校验条件，不在现有统一数据上盲目重跑。

默认来源路径统一见 [configs/paths.json](../../configs/paths.json)。`--project` 仍控制历史准备脚本中配置相对路径的基准，显式输入／输出参数优先；新来源还须符合各脚本的既定批次格式和纳入规则，不能仅改路径绕过校验。详情见 [配置说明](../../configs/README.md)。
