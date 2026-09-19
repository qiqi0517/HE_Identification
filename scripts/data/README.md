# 数据处理脚本

此目录保存数据清洗、标签整理、合并以及实验划分准备脚本；Agent 入口为 src/he_agent.py。命令从项目根目录运行。

| 脚本 | 用途 |
| --- | --- |
| merge_he_dcnn.py | 整理 HE_DCNN 图片、去重并生成二分类标签 |
| prepare_bioengineering_us.py | 校验 Bioengineering PDF 提取图片、筛选肝超声并整理标签 |
| prepare_web_us_filtered.py | 历史一次性操作：排除指定箭头来源并整理网络图片；当前数据已经处理，不应重复执行 |
| merge_processed_uuid.py | 将三个处理后来源合并，以 UUID 命名，并单独保存未标注图片 |
| merge_normal_labels.py | 将正常来源标签 0 纳入合并数据集的主标签表 |
| prepare_reference_split.py | 按患者/来源/重复组建立平衡参考池，生成固定划分；从 src/reference_split.py 拆出 |
| reference_diversity.py | 基于原划分生成来源覆盖对照和共同测试队列；从 src/ 移入 |

使用 `python scripts/data/脚本名.py --help` 查看参数。多数脚本拒绝覆盖已有输出。数据已整理好时无需再次运行这些脚本；运行 Agent 直接读取 data/unified。

划分配置已经存在时，Agent 只读取 configs/ 中的 JSON，调用 src/reference_split.py 的 validate_split 和 select_references，无需执行这两个生成脚本。生成器供重建、扩充数据或准备新实验使用，不是废弃文件。

查看参数（不生成文件、不调用 API）：

```sh
python scripts/data/prepare_reference_split.py --help
python scripts/data/reference_diversity.py --help
```
