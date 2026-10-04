# Verified limitations

1. The supplied attachment 1 has 100 MP4 videos, not the originally described five. No other video format was tested.
2. The local source data did not include precomputed 74-dimensional COVAREP MAT files. Q1 cannot reproduce trained E7a Audio features from the supplied MP4s alone. The openSMILE fallback has a different feature definition and cannot be relabeled as COVAREP-74.
3. FFmpeg/FFprobe, OpenFace, WhisperX, and complete BERT weights were absent from the tested Windows environment. No raw-video end-to-end result, all-video batch result, or production checkpoint+BERT output is claimed.
4. Q1 word-level 768-dimensional BERT embeddings differ from E7a token ID input. The adapter retokenizes words, repeats word Audio/Vision features over WordPieces, maps explicit masks, and truncates at 50 token positions. It is structurally valid but this conversion was not part of the original E7a training distribution; prediction accuracy for it is unknown.
5. Public release excludes the original videos and attachment 2 files pending a documented redistribution license and privacy review. The public manifests allow local integrity checks but do not supply data.
6. Native E7a explanation drivers remain tied to Q3's original feature format. Direct video-to-frame attribution was not verified.
7. Linux setup instructions are adapted from source documentation; execution on Linux and a fresh GitHub clone has not been verified here.
8. The public repository does not yet contain the three compact checkpoints because Git LFS upload was unavailable from the publishing host. Model inference in a fresh clone requires obtaining these weights separately; the local checkout retains verified copies.
