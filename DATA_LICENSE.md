# Code, model, and data rights

The repository's `LICENSE` applies to original project code and documentation only. It does not license datasets, third-party tools, or third-party pretrained weights.

## Compact E7a checkpoints

The three compact checkpoints contain trained E7a non-BERT tensors. Their creation did not include optimizer state or the full frozen BERT model. Their use with source data remains subject to the source data's conditions. No separate assertion about unrestricted commercial use of the weights is made here.

## Attachment 1

The local attachment contains 100 CMU-MOSEI-derived MP4 clips and a label workbook. **These files are excluded from public Git history.** The current [CMU Multimodal SDK FAQ](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/README.md?plain=1) says it cannot share original videos because of YouTube creator privacy. An earlier [CMU-MOSEI paper](https://aclanthology.org/P18-1208.pdf) contains a footnote about Creative Commons redistribution; this does not identify licenses for the individual clips in this local copy. No clip-specific redistribution authorization was supplied.

The videos can show faces, contain identifiable voices, names, subtitles, or account information. Local research use and any further sharing require a separate rights and privacy review.

## Attachment 2

The local attachment contains two feature pickles and one label workbook. No independent redistribution license was found in the supplied files. The aligned pickle also contains `raw_text` and sample IDs, so it should not be assumed anonymous. **These files are excluded from public Git history** pending a documented license and privacy review. Obtain the data from its authorized publisher or provide a locally authorized copy.

δ���ֶ����������ٷַ�����֤��������Դ��ʹ����������ԭʼ����������Ϊ׼��

## Third-party tools and models

The full [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) weights are not bundled; its model page identifies Apache-2.0. [OpenFace 2.2](https://github.com/TadasBaltrusaitis/OpenFace) has its own noncommercial research license. [COVAREP](https://github.com/covarep/covarep) has component-specific licenses. WhisperX, FFmpeg, MFA, PyTorch, and other dependencies retain their upstream terms. Obtain and inspect these from official sources before use.
