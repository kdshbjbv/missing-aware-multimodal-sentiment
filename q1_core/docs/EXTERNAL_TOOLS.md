# 外部工具与模型

本项目只依赖 `data/` 和 `Q1/` 两个顶层目录。第三方工具可以安装到系统环境，也可以放在 `Q1/external/`，但不依赖其他项目目录。

## 1. FFmpeg

确保以下命令可用：

```bash
ffmpeg -version
ffprobe -version
```

项目使用FFmpeg将视频音轨转换为16 kHz单声道WAV，并用FFprobe读取时长、帧率和音频流信息。

## 2. Montreal Forced Aligner

Version A需要MFA及英语词典/声学模型：

```bash
conda install -n temenv -y -c conda-forge montreal-forced-aligner
conda activate temenv

MFA_ROOT_DIR="$PWD/cache/mfa" mfa model download dictionary english_us_arpa
MFA_ROOT_DIR="$PWD/cache/mfa" mfa model download acoustic english_us_arpa
mfa version
```

MFA输出的TextGrid必须包含非空 `words` tier。模型与数据库缓存位于 `Q1/cache/mfa/`。

## 3. OpenFace 2.2

OpenFace是C++工具，不能用其他同维度特征替代后仍声明为OpenFace。

### 方式A：安装到系统PATH

将 `FeatureExtraction` 加入 `PATH`，YAML保持：

```yaml
paths:
  openface_feature_extraction: FeatureExtraction
  openface_library_dir: ""
```

检查：

```bash
FeatureExtraction -help
```

### 方式B：构建到Q1目录

```bash
cd Q1
mkdir -p external
git clone --depth 1 --branch OpenFace_2.2.0 \
  https://github.com/TadasBaltrusaitis/OpenFace.git external/OpenFace

cmake -S external/OpenFace -B external/OpenFace/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build external/OpenFace/build -j4
```

随后修改YAML：

```yaml
paths:
  openface_feature_extraction: external/OpenFace/build/bin/FeatureExtraction
```

OpenFace需要CMake、C++编译器、OpenCV、Boost、Eigen、dlib和视频编解码库。官方源码包可能需要额外下载CEN patch expert模型，请遵循OpenFace的模型下载说明。

Windows用户可使用OpenFace 2.2官方二进制包，并在YAML中填写相对于 `Q1/` 的路径，例如：

```yaml
paths:
  openface_feature_extraction: external/OpenFace/FeatureExtraction.exe
```

## 4. COVAREP

COVAREP原始实现通常依赖MATLAB。本项目读取已经生成的74维MAT，不会用其他音频特征冒充COVAREP。

把MAT文件放在：

```text
data/covarep/
```

即从 `Q1/` 看是：

```text
../data/covarep/
```

每个MAT应包含：

```text
features: (T, 74)
names: 74个特征名
```

复现脚本默认读取该目录，也可以通过环境变量覆盖：

```bash
COVAREP_DIR=/path/to/covarep bash scripts/reproduce_versionB.sh
```

## 5. WhisperX与模型

Version B使用WhisperX `large-v3` 和英语词级对齐模型。安装版本以 `environment/requirements-repro.txt` 和 `environment/runtime_versions.txt` 为准。

```bash
conda activate temenv
python -c "import whisperx; print(whisperx.__version__)"
```

首次运行需要下载模型。可通过YAML中的 `huggingface_home` 和 `hf_endpoint` 配置缓存位置与镜像。

## 6. GPU说明

- WhisperX和BERT可以使用CUDA；
- 默认WhisperX配置为 `float16`、batch size 8；
- OpenFace是否使用GPU取决于所安装的构建版本；
- COVAREP MAT是预先计算的输入，读取和词窗池化不需要GPU。

外部工具不可用时，程序应明确报错或记录缺失状态，不会伪造特征。
