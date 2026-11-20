# HFFF

Implementation of **Hierarchical Fine-grained Fusion with Temporally Anchored Queries for Multimodal Sentiment Analysis**. HFFF uses text queries and temporally anchored audio and video queries to summarize each modality before fusion. The implementation covers CMU-MOSI, CMU-MOSEI and CH-SIMS.

## Setup

Run all commands from the repository root. Install uv and make FFmpeg and ffprobe available on your PATH. The paper used a single NVIDIA RTX 5090 with 32 GB of memory, Python 3.14, PyTorch 2.14.1 and Lightning 2.6.6. The lockfile includes the GPU runtime used by the InsightFace detector. Initial preprocessing and training require access to the detector and pretrained models, or locally cached copies. Backbone identifiers are in [config.py](config.py).

```bash
uv sync --locked
```

## Prepare a dataset

Obtain the original video clips and sentiment annotations through the [MMSA project](https://github.com/thuiar/MMSA). Place the dataset directory alongside the repository, using this layout for CH-SIMS. Use the directory name `MOSI` or `MOSEI` for the English datasets.

```text
parent/
├── hfff/                      # this repository
└── datasets/
    └── SIMS/
        ├── label.csv
        └── Raw/<video_id>/<clip_id>.mp4
```

Keep video and clip IDs as strings, including leading zeros. The English label files have column headers. The CH-SIMS file is headerless with columns in the order `video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode`. The loader uses the supplied train, valid and test splits.

```bash
uv run python prepare/split_av.py --dataset SIMS
uv run python prepare/cache_face.py --dataset SIMS
uv run python prepare/cache_face_timing.py --dataset SIMS
```

The splitter writes 16 kHz WAV files and silent videos alongside the raw clips. The next two commands write face crops, detection metadata and frame timestamps to `data_cache/SIMS/` in the repository. For the whole-frame comparison, run `uv run python prepare/cache_frame.py --dataset SIMS` after splitting and pass `--visual-input frame` during training.

## Train and evaluate

```bash
uv run python main.py --dataset SIMS
uv run python main.py --dataset MOSI --seed 0
uv run python main.py --dataset SIMS --ablation-mode wo_temporal_anchor --seed 0 1 2
```

The default is the full HFFF model with face crops and seeds 0, 1 and 2. Use `python main.py --help` in the project environment for the implemented ablations. The paper's single-GPU settings are:

| Dataset | Training batch size | Gradient accumulation | Maximum epochs |
| --- | ---: | ---: | ---: |
| CMU-MOSI, CH-SIMS | 2 | 8 | 16 |
| CMU-MOSEI | 1 | 32 | 3 |

Early stopping and checkpoint selection monitor validation MAE. Each run tests its best checkpoint. Find the weights under `mlruns/<experiment_id>/<run_id>/artifacts/checkpoints/` and predictions, features and the t-SNE plot under the adjacent `test_outputs/` directory.
