# OpenFace视觉特征

本项目使用OpenFace 2.2生成逐帧人脸行为特征，并按词级时间窗聚合成定长视觉序列。

## 1. 原始CSV

每条视频由OpenFace `FeatureExtraction` 生成一个CSV。重新运行时，文件位于对应版本的工作目录：

```text
work/versionA/openface/<safe_sample_name>/<clip_id>.csv
work/versionB/openface/<safe_sample_name>/<clip_id>.csv
```

`sample_id` 中的 `$_$` 会转换为安全文件名分隔符 `__`。

构建35维表示前，代码会审计所有CSV中的 `AU*_r` 和 `AU*_c` 字段。只有全部CSV字段一致且实际数量为17个强度字段与18个出现字段时才继续处理，不补零、不使用随机映射，也不通过PCA或线性层强制扩维。

## 2. 35维字段

固定顺序为：

```text
AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r,
AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r,
AU25_r, AU26_r, AU45_r,
AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c,
AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c,
AU25_c, AU26_c, AU28_c, AU45_c
```

其中：

- `AU*_r`：动作单元强度；
- `AU*_c`：动作单元是否出现，逐帧值通常为0或1；
- 词窗内对 `AU*_c` 取均值后，数值表示该动作单元的激活帧比例。

该35维表示是项目定义的OpenFace特征，不是FACET 35维，也不是官方CMU-MOSEI视觉特征。

## 3. 有效帧

视觉帧必须同时满足：

```text
success > 0
confidence >= 0.8
```

无效帧不会参与词窗均值。逐帧缓存保存时间戳、35维特征、有效标志、置信度及字段名。

典型缓存位置：

```text
<work-dir>/features/vision_35/<safe_sample_name>.npz
```

## 4. 词窗池化

对词级区间 `[start_i, end_i)`，选择满足以下条件的帧：

```python
selected = (
    (frame_times >= start_i)
    & (frame_times < end_i)
    & valid_frames
)
```

若存在有效帧：

```text
vision[i] = selected_features.mean(axis=0)
vision_mask[i] = 1
vision_frame_counts[i] = 有效帧数
```

若真实词窗内没有有效帧：

```text
vision[i] = zeros(35)
vision_mask[i] = 0
vision_frame_counts[i] = 0
sequence_mask[i] = 1
```

这表示视觉观测缺失，不是序列Padding。

## 5. 22维兼容字段

结果中保留 `vision_22`：

```text
17维 AU intensity
+ 3维头部旋转 pose_Rx, pose_Ry, pose_Rz
+ 2维注视角 gaze_angle_x, gaze_angle_y
= 22维
```

`vision_22` 仅用于兼容和对照，主视觉特征是 `vision`。

## 6. 最终结构

`results/versionA.pkl` 和 `results/versionB.pkl` 中每条样本包含：

```python
{
    "vision": ndarray,              # (50, 35)
    "vision_22": ndarray,           # (50, 22)
    "vision_mask": ndarray,         # (50,)
    "vision_frame_counts": ndarray, # (50,)
    "timestamps": ndarray,          # (50, 2)
    "sequence_mask": ndarray,       # (50,)
}
```

读取示例：

```python
import pickle

with open("results/versionA.pkl", "rb") as handle:
    dataset = pickle.load(handle)

sample = dataset["samples"][0]
print(sample["sample_id"])
print(sample["vision"].shape)
print(sample["vision_22"].shape)
print(int(sample["vision_mask"].sum()))
```

## 7. 当前结果统计

Version A：85条视觉全部有效、10条部分缺失、5条完全缺失。

Version B：83条视觉全部有效、11条部分缺失、6条完全缺失。

所有样本均保留。视觉缺失通过零向量、`vision_mask` 和 `vision_frame_counts` 显式表达。
