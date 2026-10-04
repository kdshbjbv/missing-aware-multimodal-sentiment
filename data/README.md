# Local data layout

The data directories are local inputs and are ignored by Git. Public manifests record their file sizes and SHA-256 hashes without including the media or feature content.

## Where to obtain and place data

The upstream dataset is [CMU-MOSEI](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), whose computational sequences are distributed through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). The SDK states that it does not distribute the original YouTube videos. **The exact 100-clip subset and the project-specific `aligned_50.pkl` / `unaligned_50.pkl` files are not identified as downloads at those upstream links.** Obtain the exact project attachments from a source that is authorized to provide them. A different MOSEI release may have different files and will not pass the included hashes or necessarily work with this model.

From the repository root, place the complete video directory and its `label-100.xlsx` under `data/attachment1/`, and place `aligned_50.pkl`, `unaligned_50.pkl`, and `label.xlsx` directly under `data/attachment2/`. Preserve filenames and file bytes. The resulting layout must be:

```text
data/
  attachment1/MOSEI数据集部分原始视频-100条/<video_id>/<clip_id>.mp4
  attachment1/MOSEI数据集部分原始视频-100条/label-100.xlsx
  attachment2/aligned_50.pkl
  attachment2/unaligned_50.pkl
  attachment2/label.xlsx
  manifests/attachment1_manifest.csv
  manifests/attachment2_manifest.csv
```

Run the two verification commands below before training or inference. They require exact path, file-size, and SHA-256 matches. The three model weights are downloaded automatically by Git LFS during a normal clone; if LFS smudge was skipped, run `git lfs pull`.

Attachment 1 has **100 MP4 files plus one label workbook**, totaling 103,633,174 bytes. They provide the Q1 raw-video input, official-text labels for Version A, and source timestamps for feature alignment. `video_id` is the parent folder, `clip_id` the MP4 stem, and Q1 forms `sample_id` as `video_id$_$clip_id`. Version B instead obtains English text and word times with WhisperX, without using the workbook labels.

Attachment 2 has **three files**, totaling 3,891,269,722 bytes. Its original pickle splits are retained: train 3,395, valid 728, test 727. Class 0 is Negative, 1 Neutral, 2 Positive. The train class counts are 967/758/1,670, valid 206/184/338, and test 207/158/362. Regression is a continuous sentiment score. The split is stored in the pickle; this project does not regenerate it. The aligned data contains `text_bert=[N,3,50]` int64, Audio `[N,50,74]` float64, Vision `[N,50,35]` float64, labels, IDs, raw text, and an additional `[N,50,768]` `text` feature. The unaligned Audio/Vision arrays use 500 positions and also include length fields. E7a training and inference use the aligned version.

The three `text_bert` channels are BERT input IDs, attention mask, and token type IDs. E7a derives valid positions from attention, excludes CLS and SEP from Audio/Vision applicability, and treats zero Audio/Vision vectors as unobserved. Q1 instead stores per-word BERT vectors and explicit Text/Audio/Vision masks. `src/data/q1_to_e7a_adapter.py` retokenizes Q1 words and maps each WordPiece back to the source word's Audio/Vision features and masks; it records truncation at 50 positions. This is a format conversion, not a claim that Q1 and attachment 2 features were extracted identically.

Verify local data after obtaining it lawfully:

```bash
python scripts/verify_dataset.py --data_dir data/attachment1 --manifest data/manifests/attachment1_manifest.csv
python scripts/verify_dataset.py --data_dir data/attachment2 --manifest data/manifests/attachment2_manifest.csv
```

Do not load pickle files from untrusted sources. Media and text can reveal faces, voices, names, subtitles, and account identifiers. See `DATA_LICENSE.md`.
