# 项目配置

## 资源路径：paths.json

[paths.json](paths.json) 集中维护主 Agent 和数据准备／分析脚本原来写死的资源路径。修改现有 `paths` 对象中的字段即可，例如把 `"dataset": "data/unified"` 改成 `"dataset": "/Volumes/Research/new_dataset"`。不要把 API key 的值写入此文件；`api_keys` 只保存密钥文件的位置。

| 字段 | 用途 |
| --- | --- |
| `dataset` | 主 Agent 和参考划分生成器的默认统一数据集 |
| `default_split` | 推理默认划分；保留 `null` 表示仍由 `--split` 指定 |
| `runs` / `threshold_sweeps` | 推理／离线阈值分析的输出父目录，程序在其下新建时间目录 |
| `baseline_prompt` / `medical_prompt` / `review_prompt` | 三类提示词文件 |
| `medical_sources` | 医学提示词来源和哈希记录，搬迁时须与提示词配套 |
| `api_keys` | 本地 API 密钥 JSON 的位置 |
| `base_split` / `split_output` | 来源优化生成器的默认基础划分／普通生成器的默认输出文件 |
| `bioengineering_source` / `bioengineering_extracted` | 原始补充材料／已提取图像目录 |
| `processed_he` / `processed_bio` / `processed_web` | 合并脚本读取的三个已整理来源；HE 路径也用于跨来源查重 |
| `web_source` / `web_staging` | 网页数据整理脚本的来源／临时准备目录；目标为 `processed_web` |

解析规则：

- 配置中的相对路径以**代码项目根目录**为基准，支持绝对路径和 `~`。配置文件放到别处，也不改变这个基准。历史数据脚本的 `--project` 可覆盖此基准。
- 显式 `--dataset`、`--split`、`--prompt`、`--output` 等参数优先于配置；CLI 相对路径仍相对当前终端目录。
- 可复制完整 JSON 为 `configs/paths.local.json`，再用 `HE_PATHS_CONFIG=configs/paths.local.json python src/he_agent.py ...` 切换整套路径。相对的配置文件名也按代码项目根目录查找；`.local.json` 已被 Git 忽略。
- 若经常使用同一划分，可将 `default_split` 设为 `configs/reference_split_balanced.json`；它会对所有模式生效。默认仍为 `null`，保持旧命令行为。
- 数据准备中原本必填的 `--source`、`--output`、`--unlabeled-output`、`--dataset` 等仍须显式提供；已有输出不允许覆盖。没有额外引入静默写入目标。

更换数据源后，目录内仍须有 `images/`、`labels/labels.txt` 和 `labels/manifest.jsonl`，且路径、标签与图片哈希一致。配置只解决**资源位置**，不会将任意原始数据自动转成统一格式。新增／替换图片必须生成匹配的新划分；只搬迁同一批图片时可继续使用成员和哈希一致的旧划分。医学提示词搬迁也保留现有哈希校验。

新运行的 `run.json` 记录 `path_config`（配置位置、哈希和解析后的默认路径），并继续记录实际数据集、提示词、划分哈希和样本快照。历史 run、split、manifest 内的溯源路径不批量改写；`experiments/` 的独立历史原型仍使用各自配置。

## 本地 API 密钥


在 `configs/api_keys.local.json` 中填写密钥：

```json
{
  "ZHIPU_API_KEY": "",
  "OPENAI_API_KEY": "",
  "DEEPSEEK_API_KEY": ""
}
```

将对应的空字符串替换为实际key，不加 `Bearer ` 前缀，不要把key填到example文件。此本地文件已由 `.gitignore` 的 `configs/*.local.*` 忽略；首次创建权限为0600，仅当前用户读写。程序不把密钥写入run配置、预测或其他实验快照。

GLM使用ZHIPU_API_KEY，GPT使用OPENAI_API_KEY，DeepSeek使用DEEPSEEK_API_KEY。读取优先级：非空环境变量 → 这个本地JSON文件。空/仅空白环境变量会回退文件；若终端仍有旧key，先 `unset ZHIPU_API_KEY` 、`unset OPENAI_API_KEY` 或 `unset DEEPSEEK_API_KEY`，即可使用文件中的key。

填写后运行原来的命令即可，无需每次打开终端重新输入。密钥文件默认位置可通过 `paths.json` 的 `api_keys` 调整，不受当前工作目录影响。无需安装新依赖，`.env` 仍不会自动读取。dry-run和离线阈值脚本不读取密钥。

`api_keys.example.json` 是可提交的空模板；在新克隆的项目中，若没有local文件，可用下面的命令复制并设置权限。已有local文件时不要覆盖。

```sh
cp -n configs/api_keys.example.json configs/api_keys.local.json
chmod 600 configs/api_keys.local.json
```

其余实验 JSON 为数据划分；路径配置与划分文件中均不填写密钥。

DeepSeek在原JSON对象内新增 `"DEEPSEEK_API_KEY": "你的key"` 即可；保留已填的GLM/GPT字段。接口使用官方 `deepseek-flash` 视觉模型，命令与比较方法见[实验指南](../docs/experiment-guide.md)。

## 划分文件

新实验显式选择 `reference_split_balanced.json`；其余 split 和 cohort 用于历史复现及旧新配对，不删除或覆盖。参考数、来源配额与队列说明统一见[实验指南](../docs/experiment-guide.md#参考池与实际参考数)。
