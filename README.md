# Missing-aware multimodal sentiment analysis

This repository organizes the Q1 raw-video feature pipeline and the frozen E7a sentiment model. It retains the original model implementation in `e7a_core/` and the Q1 implementation in `q1_core/`. The public wrappers in `scripts/` connect them through an explicit adapter.

```text
MP4 → FFmpeg WAV → WhisperX English words and timestamps
    → COVAREP-74 Audio + OpenFace-35 Vision + BERT word features
    → word-window alignment (50 Q1 positions)
    → WordPiece retokenization and mask transfer
    → frozen BERT + three E7a members
    → equal mean of original raw logits, argmax + mean regression
```

**Release scope:** the public Git repository contains code, tests, and integrity manifests. The three compact E7a checkpoints are hash-verified in the local checkout, but their Git LFS objects could not be uploaded from this host, so they are not present in the public repository yet. The 100 local CMU-MOSEI videos and the three attachment 2 files are deliberately excluded from Git because their redistribution rights were not established. Their local copies are unchanged and match the original files by SHA-256. See [DATA_LICENSE.md](DATA_LICENSE.md) and [data/README.md](data/README.md).

## What was actually checked

- Attachment 1 contains 100 MP4 clips and one workbook, not five clips. Only MP4 appears in this source set. Format/codec probing could not be repeated here because FFprobe is unavailable on the tested Windows machine.
- Attachment 2 has the original train/valid/test splits of 3,395/728/727. Its aligned tensors are `text_bert=[N,3,50]`, `audio=[N,50,74]`, and `vision=[N,50,35]`. The unaligned Audio/Vision arrays have 500 time positions.
- All three compact checkpoint hashes match `e7a_core/checkpoint_manifest.json`; all three state dicts load into E7a with a test encoder and execute a CPU forward pass.
- The Q1→E7a adapter and synthetic CPU tests pass. Full raw-video inference was not verified: the supplied data has no COVAREP MAT files and the tested environment lacks FFmpeg, OpenFace, WhisperX, and the BERT weights.

## Layout

| Path | Purpose |
|---|---|
| `q1_core/` | Original Q1 code and reference documentation |
| `e7a_core/` | Original E7a computation, trainer, evaluator, explanation modules |
| `src/data/q1_to_e7a_adapter.py` | Q1 word sequence to E7a token sequence |
| `scripts/` | Portable public entry points and validation |
| `checkpoints/` | Three compact non-BERT member weights in the local checkout; public upload pending |
| `data/manifests/` | Local data hashes; no data content |
| `data/attachment1`, `data/attachment2` | User-supplied local inputs, excluded from Git |

## Installation

Python 3.10 is the reference version used in the source projects. Local verification here used **Windows, Python 3.11.4, PyTorch 2.1.2+cpu, Transformers 4.37.1, NumPy 1.24.3** for the tests that did not require the full model. Linux execution was not tested. CUDA is optional for the synthetic tests and E7a inference, but the original WhisperX `large-v3` extraction route is designed for a CUDA GPU; CPU is likely slow. The original Q1 environment references CUDA 12.6 and PyTorch 2.8.0.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python scripts/verify_checkpoints.py --checkpoint_dir checkpoints  # after placing the three local weights
```

For the full Q1 extractor, create a separate compatible Python 3.10 environment and install `q1_core/environment/requirements-repro.txt`; its pinned PyTorch/WhisperX stack differs from the lightweight test environment. On Windows use PowerShell activation as above; on Linux use the `source` command. The external executables and model downloads still need separate installation.

The full video route additionally requires FFmpeg/FFprobe, OpenFace 2.2 `FeatureExtraction`, WhisperX, COVAREP 74-dimensional MAT files, and the English BERT model. See [raw-video setup](docs/raw_video_pipeline.md) and the original [external tool notes](q1_core/docs/EXTERNAL_TOOLS.md). Q1 accepts the supplied MP4 tree; AVI/MOV/MKV and other languages were not tested. Audio absence is reported by the Q1 processing stages. The recommended video length is the short utterance clips represented by this dataset; longer input may be truncated to 50 positions.

Download [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) through the official Hugging Face tooling, then place the full model and tokenizer directory in `pretrained/bert-base-uncased/`. The default configs use `local_files_only: true`. For an online run, change `model.pretrained_name` to `google-bert/bert-base-uncased` and `local_files_only` to `false` in the model config. Download WhisperX and MFA models from their upstream sources according to Q1's documentation. Large third-party caches are not committed.

## Local data checks

After obtaining a lawfully distributable copy of the data in the structure described in [data/README.md](data/README.md):

```bash
python scripts/verify_dataset.py --data_dir data/attachment1 --manifest data/manifests/attachment1_manifest.csv
python scripts/verify_dataset.py --data_dir data/attachment2 --manifest data/manifests/attachment2_manifest.csv
python scripts/prepare_dataset.py --input data/attachment2 --output_dir outputs/attachment2_prepared --config configs/e7a_train.yaml
```

The validation script verifies every filename, size, and SHA-256. It expects 100 videos plus the attachment 1 workbook, and the three attachment 2 files. No source content is printed.

## Extract and predict

Supply precomputed COVAREP MAT files matching the video sample IDs in `precomputed/covarep/`. The current source directory did not contain those files. An openSMILE fallback with a different acoustic feature definition is not silently substituted for the trained E7a input.

```bash
python scripts/extract_features.py --input_dir data/attachment1 --output_dir outputs/attachment1_features --config configs/feature_extraction.yaml --covarep_dir precomputed/covarep
python scripts/extract_features.py --input_video data/attachment1/MOSEI数据集部分原始视频-100条/VIDEO_ID/CLIP_ID.mp4 --output_dir outputs/one/features --config configs/feature_extraction.yaml --covarep_dir precomputed/covarep
python scripts/predict_video.py --input_video data/attachment1/MOSEI数据集部分原始视频-100条/VIDEO_ID/CLIP_ID.mp4 --output_dir outputs/one --feature_config configs/feature_extraction.yaml --model_config configs/e7a_inference.yaml --covarep_dir precomputed/covarep
python scripts/predict_video.py --input_dir data/attachment1 --output_dir outputs/attachment1 --feature_config configs/feature_extraction.yaml --model_config configs/e7a_inference.yaml --covarep_dir precomputed/covarep
```

The commands above are supported entry points but were **not** run end to end on this host. The wrapper fails explicitly when prerequisites are absent. Existing Q1 output can be sent directly to `python scripts/predict_features.py --features /path/to/versionB.pkl --output_dir outputs/prediction --config configs/e7a_inference.yaml` once BERT is installed. Predictions use the mean of the three members' **原始logits** (raw logits), zero class bias `[0,0,0]`, then argmax. Regression uses the mean of three regression outputs. The CSV contains ID, class, regression, three raw logits, probabilities, observed Text/Audio/Vision flags, and truncation.

## Train and evaluate E7a

The wrappers call the retained E7a trainer/evaluator. They preserve the source pickle splits and do not use the test split for model selection.

```bash
python scripts/train_e7a.py --data_dir data/attachment2 --config configs/e7a_train.yaml --seed 42
python scripts/train_e7a.py --data_dir data/attachment2 --config configs/e7a_train.yaml --seed 17
python scripts/train_e7a.py --data_dir data/attachment2 --config configs/e7a_train.yaml --seed 2026
python scripts/evaluate_e7a.py --data_dir data/attachment2 --checkpoint checkpoints/e7a_seed42_compact.pt --split valid --config configs/e7a_inference.yaml
python scripts/evaluate_e7a.py --data_dir data/attachment2 --checkpoint checkpoints/e7a_seed42_compact.pt --split test --config configs/e7a_inference.yaml
```

The original E7a architecture includes a frozen BERT encoder, Text/Audio/Vision projections, shared cross-attention compensator with a learnable NULL token, missing-aware gate, two-layer Transformer encoder, mask-aware pooling, classification and regression heads, and multitask reconstruction loss. The trainer's missing-modality simulation uses continuous blocks. The core calculation was not edited in this release.

## Explanations and limitations

Original Q3 explanation implementations remain in `e7a_core/src/explain.py` and `explain_classification.py`, with task-specific drivers in `e7a_core/`. They cover modality Shapley, classification raw logits, regression, Integrated Gradients, occlusion faithfulness, gate weights, and missingness diagnostics for the original Q3 data flow. A generic raw-video explanation entry point and reliable mapping back to frames were **not** verified, so this repository makes no claim that `predict_video.py` produces explanations. See [limitations](docs/limitations.md).

The code license is MIT. No dataset rights are granted by that license. Cite the original CMU-MOSEI paper and this repository when appropriate. Third-party pretrained models and extraction tools have their own terms.

