# 三模态特征提取与词级时序对齐流程

## 1. 目标

从英文短视频构建统一的词级三模态序列：

```text
text       (50, 768)
audio      (50, 74)
vision     (50, 35)
timestamps (50, 2)
```

每个时序位置对应原视频中的一个词级区间 `[start, end)`。

## 2. 共享模块

两条流程复用以下模块：

1. FFmpeg将MP4音轨转换为16 kHz单声道WAV；
2. BERT-base-uncased生成768维词级文本特征；
3. COVAREP MAT转换为74维逐帧语音特征；
4. OpenFace 2.2从MP4生成逐帧视觉特征；
5. 按词级时间窗池化语音和视觉帧；
6. 统一长度为50并生成Mask和帧计数；
7. 保存PKL、CSV摘要和JSON验证报告。

## 3. Version A

```text
MP4 + label-100.xlsx + COVAREP MAT
  ├─ FFmpeg → WAV
  ├─ Excel官方文本 → MFA → 词级时间戳
  ├─ 官方词 → BERT
  ├─ COVAREP MAT → 逐帧音频特征
  ├─ MP4 → OpenFace → 逐帧视觉特征
  ├─ 按MFA时间窗池化三模态
  ├─ WhisperX/Pyannote/COVAREP缺失文本审计
  └─ results/reproduced/versionA.pkl
```

Version A的语义主体始终来自Excel官方文本。WhisperX时间轴只在严格证明官方文本存在遗漏时用于建立缺失位置；Pyannote和COVAREP用于审计有声但未被文本覆盖的区间。缺失位置不会伪造BERT文本向量。

当前保存的参考结果位于 `results/versionA.pkl`。

## 4. Version B

```text
MP4 + COVAREP MAT
  ├─ FFmpeg → WAV
  ├─ WAV → WhisperX → ASR文本和词级时间戳
  ├─ WhisperX词 → BERT
  ├─ COVAREP MAT → 逐帧音频特征
  ├─ MP4 → OpenFace → 逐帧视觉特征
  ├─ 按WhisperX时间窗池化三模态
  └─ results/reproduced/versionB.pkl
```

Version B不读取标签Excel，所有标签字段均为 `None`，并设置 `labels_available=False`。

当前保存的参考结果位于 `results/versionB.pkl`。

## 5. 文本特征

BERT输入词可能被拆成多个WordPiece。代码利用tokenizer的 `word_ids` 将同一原词对应的最后一层隐状态取均值，得到一个768维词向量。

Version A编码官方文本词；Version B编码WhisperX识别词。两者的BERT缓存相互隔离。

## 6. 音频特征

COVAREP MAT中包含：

```text
features: (T, 74)
names: 74个字段名
```

默认帧移为0.01秒。对第 `i` 个词窗：

```python
selected = (audio_times >= start_i) & (audio_times < end_i)
audio[i] = selected_features.mean(axis=0)
```

无帧时：

```text
audio[i] = zeros(74)
audio_mask[i] = 0
audio_frame_counts[i] = 0
```

两条数字静音样本没有COVAREP MAT，经WAV和原视频音轨检查确认静音后保留，音频特征及Mask均为零。

## 7. 视觉特征

OpenFace CSV必须经过字段审计。主视觉表示为：

```text
17维 AU intensity (`AU*_r`)
+ 18维 AU presence (`AU*_c`)
= 35维
```

有效帧规则：

```text
success > 0 and confidence >= 0.8
```

对每个词窗中的有效帧逐维取均值。无有效帧时，`vision`、`vision_mask` 和 `vision_frame_counts` 均为零。旧22维OpenFace表示保存在 `vision_22` 中。

详细字段见 [`OPENFACE_VISUAL_FEATURES.md`](OPENFACE_VISUAL_FEATURES.md)。

## 8. 统一对齐

第 `i` 个位置满足：

```text
words[i]
timestamps[i] = [start_i, end_i)
text[i]
audio[i]
vision[i]
```

三种模态不是因为维度相同而被视为对齐，而是共同引用同一个词级时间窗。

## 9. 缺失值和定长

`MAX_LEN=50`。

- `L < 50`：尾部补零；
- `L = 50`：原样保留；
- `L > 50`：保留前50个位置，设置 `truncated=True` 并保存 `original_word_count`。

真实位置缺少某模态与Padding不同：

| 状态 | `sequence_mask` | 模态Mask | 模态特征 |
|---|---:|---:|---|
| 有真实词且模态存在 | 1 | 1 | 池化特征 |
| 有真实词但模态缺失 | 1 | 0 | 全零 |
| Padding | 0 | 0 | 全零 |

Padding词为空字符串，时间戳为 `[0,0]`。

## 10. 中间目录

复现脚本分别使用：

```text
work/versionA/
work/versionB/
```

典型结构：

```text
audio/                    WAV和MAT缓存
mfa/corpus/               MFA输入
mfa/aligned/              TextGrid
mfa/word_timestamps/      MFA词级JSON
asr/whisperx/             WhisperX结果
openface/                 OpenFace原始CSV
features/text/            Version A文本特征
features/text_whisperx/   Version B文本特征
features/audio/           COVAREP逐帧缓存
features/vision/          旧22维视觉缓存
features/vision_35/       35维视觉缓存
```

## 11. 运行

```bash
conda activate temenv
cd Q1

bash scripts/reproduce_versionB.sh
bash scripts/reproduce_versionA.sh
```

检查现有结果：

```bash
python scripts/validate_results.py
python scripts/inspect_pkl.py results/versionA.pkl
python scripts/inspect_pkl.py results/versionB.pkl
```

## 12. 当前验证结果

```text
Version A: 100/100 samples valid
Version B: 100/100 samples valid
text/audio/vision dimensions: 768/74/35
NaN/Inf: 0
```

验证内容包括张量形状、数值有限性、时间序列Mask、模态Mask、帧计数、Padding、样本唯一性和标签语义。
