# 数据格式

本项目提供两个词级对齐数据文件：

```text
results/versionA.pkl
results/versionB.pkl
```

二者使用相同的张量结构。Version A使用Excel官方文本和MFA时间轴并包含情感标签；Version B使用WhisperX文本和时间轴，不包含情感标签。

## 1. 顶层结构

文件由Python `pickle` 保存：

```python
{
    "samples": [sample_0, sample_1, ..., sample_99],
    "meta": {...}
}
```

读取示例：

```python
import pickle

with open("results/versionA.pkl", "rb") as handle:
    dataset = pickle.load(handle)

print(dataset["meta"])
print(len(dataset["samples"]))
```

## 2. 核心张量

每条样本定长为50：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `text` | `(50, 768)` | BERT词级特征 |
| `audio` | `(50, 74)` | COVAREP词窗均值 |
| `vision` | `(50, 35)` | OpenFace 17维AU强度 + 18维AU出现率 |
| `vision_22` | `(50, 22)` | 兼容保留的17维AU强度 + 3维头姿 + 2维注视角 |
| `timestamps` | `(50, 2)` | 原视频中的词窗 `[start, end)`，单位为秒 |
| `sequence_mask` | `(50,)` | 真实时序位置为1，Padding为0 |
| `text_mask` | `(50,)` | 该位置是否具有可编码文本 |
| `audio_mask` | `(50,)` | 词窗内是否存在COVAREP帧 |
| `vision_mask` | `(50,)` | 词窗内是否存在有效OpenFace帧 |
| `audio_frame_counts` | `(50,)` | 参与语音池化的帧数 |
| `vision_frame_counts` | `(50,)` | 参与视觉池化的有效帧数 |

张量类型以PKL实际值为准，浮点特征通常为 `float32`，Mask为整数或布尔兼容数组，帧计数为整数数组。

## 3. 样本标识与文本字段

| 字段 | 含义 |
|---|---|
| `sample_id` | 唯一样本编号，格式为 `video_id$_$clip_id` |
| `video_id` | 视频组编号 |
| `clip_id` | 片段编号 |
| `raw_text` | 当前版本采用的原始文本字符串 |
| `words` | 长度50的词列表，Padding位置为空字符串 |
| `valid_length` | 真实时序位置数，范围0–50 |
| `original_word_count` | 截断前的词或审计位置总数 |
| `duration` | 原视频时长，单位为秒 |
| `truncated` | 是否因超过50个位置而截断 |

Version B还包含 `asr_text`、`whisperx_status`、`whisperx_model` 和 `whisperx_alignment_model` 等ASR字段。Version A还包含官方文本、MFA以及缺失文本审计相关字段。

## 4. 标签字段

Version A：

```text
regression_label
classification_label
annotation
```

三者来自标签Excel，并与 `sample_id` 一一对应。

Version B：

```text
labels_available = False
regression_label = None
classification_label = None
annotation = None
```

Version B中的占位字段不能用于监督训练，也不能解释为中性标签。

## 5. 对齐语义

对任意有效位置 `i`：

```text
words[i]
timestamps[i] = [start_i, end_i)
text[i]
audio[i]
vision[i]
```

以上字段描述同一个词级时间窗。语音和视觉均使用：

```text
start_i <= frame_time < end_i
```

视觉帧还必须满足：

```text
success > 0 and confidence >= 0.8
```

## 6. 缺失模态与Padding

真实词窗内缺少某模态时，该位置仍是有效时序位置：

```text
sequence_mask[i] = 1
modality_mask[i] = 0
modality_feature[i] = 0
modality_frame_counts[i] = 0
```

Padding位置：

```text
words[i] = ""
timestamps[i] = [0, 0]
sequence_mask[i] = 0
text_mask[i] = audio_mask[i] = vision_mask[i] = 0
所有模态特征为0
```

因此不能只根据特征是否为零判断Padding，应优先使用 `sequence_mask`。

## 7. 视觉特征顺序

`vision` 的35个字段顺序固定为：

```text
AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r,
AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r,
AU25_r, AU26_r, AU45_r,
AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c,
AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c,
AU25_c, AU26_c, AU28_c, AU45_c
```

`AU*_r` 的词窗均值表示平均动作强度；`AU*_c` 的词窗均值表示激活帧比例。该表示不是FACET 35维。

## 8. Manifest和统计文件

`versionA_manifest.csv` 和 `versionB_manifest.csv` 保存源视频相对路径、媒体元信息和样本顺序。`video_path` 为相对路径，便于迁移项目。

`versionA_feature_summary.csv` 和 `versionB_feature_summary.csv` 保存每条样本的有效长度、模态状态、维度和处理状态，可在不读取PKL的情况下进行快速检查。

## 9. 验证

```bash
python scripts/validate_results.py
```

验证器检查样本数量、张量形状、NaN/Inf、Mask、帧计数、Padding和标签语义。
