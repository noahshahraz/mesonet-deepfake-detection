# MesoNet — Deepfake Detection (reproduction)

A from-scratch PyTorch reproduction of **MesoNet** (Afchar, Nozick, Yamagishi & Echizen,
*MesoNet: a Compact Facial Video Forgery Detection Network*, WIFS 2018,
[arXiv:1809.00888](https://arxiv.org/abs/1809.00888)). Two deliberately tiny CNNs — **Meso-4** and
**MesoInception-4** (~28k parameters each) — classify a face image as **real** or **forged**.

Trained locally on an Apple Silicon MacBook Pro (M-series, 24 GB) using the PyTorch **MPS** backend.

## What this project does
1. **Reproduces** the paper's headline numbers on **FaceForensics++** (~98% accuracy on Deepfakes,
   ~95% on Face2Face).
2. **Tests generalization** — takes the FaceForensics++-trained model and evaluates it, unchanged,
   on other deepfake datasets to measure how far performance transfers.

## Method in brief
Both networks work at a *mesoscopic* scale — between raw pixel noise (destroyed by video
compression) and high-level semantics — where forgery artifacts survive.

- **Meso-4:** four `Conv → ReLU → BatchNorm → MaxPool` blocks (8, 8, 16, 16 filters; the last pool
  is 4×4), then `Dropout → Dense(16) → LeakyReLU → Dropout → Dense(1)`.
- **MesoInception-4:** the first two conv blocks are replaced by Inception-style modules with
  parallel branches using **dilated** 3×3 convolutions (rates 1–3), concatenated; the rest matches
  Meso-4.

See [`src/models/`](src/models). Both nets pass `pytest -q` and come in under 50k parameters.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # model smoke tests should pass immediately
```

## Data
Three datasets, all normalised to `data/<name>/<split>/{real,fake}/*.jpg`. See
[`scripts/download_data.md`](scripts/download_data.md). Start with **OpenForensics** (no access
form); request **FaceForensics++** access in parallel.

| Dataset | Role | Images | Fake type |
|---|---|---|---|
| OpenForensics (`manjilkarki`) | quick baseline | 190,335 | forged faces, in-the-wild |
| FaceForensics++ | **paper reproduction** | ~15–20k face crops | face-swap + reenactment |
| 140k Real/Fake (`xhlulu`) | generalization | 140,002 | StyleGAN synthetic |

## Reproduce (one line)
```bash
# train then evaluate Meso-4 end-to-end using the default config
python -m src.train --config configs/default.yaml && \
python -m src.eval  --config configs/default.yaml --checkpoint checkpoints/best.pth
```

## Results

### Baseline — OpenForensics (NOT paper-comparable)
Full 190k-image Kaggle set (`manjilkarki/deepfake-and-real-images`), full train split (140,002
images), best-val-AUC checkpoint, threshold 0.5. Both models reach val AUC ≈ 0.990; the test
split is markedly harder than val (val acc ~0.93 vs test ~0.87) — a known distribution quirk of
this dataset, reported as-is.

| Model | Test acc. | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Meso-4 | 0.871 | 0.953 | 0.892 | 0.846 | 0.868 |
| MesoInception-4 | 0.871 | 0.952 | 0.882 | 0.860 | 0.871 |

### Reproduction — paper vs. this repo (FaceForensics++)
_Only FaceForensics++ is comparable to the paper._ Filled in during the build phase (T16).

| Method (Meso-4) | Paper acc. | Mine acc. | Mine AUC |
|---|---|---|---|
| Deepfakes | ~0.98 | _TBD_ | _TBD_ |
| Face2Face | ~0.95 | _TBD_ | _TBD_ |

### Generalization — one FF++-trained model, evaluated across datasets
_Not paper-comparable; the point is how much accuracy shifts off the training distribution._
Filled in during Phase 3 (T18–T19).

| Eval dataset | Meso-4 acc. | Meso-4 AUC | Xception acc. |
|---|---|---|---|
| FaceForensics++ (in-domain) | _TBD_ | _TBD_ | _TBD_ |
| OpenForensics | _TBD_ | _TBD_ | _TBD_ |
| 140k (StyleGAN) | _TBD_ | _TBD_ | _TBD_ |

## Repo layout
```
configs/default.yaml      all hyperparameters (nothing hardcoded)
src/models/               Meso4, MesoInception4
src/data/                 dataset + transforms
src/utils/                device (MPS-first), seed, config, metrics
src/train.py  src/eval.py entry points
scripts/                  data download notes, FF++ face extraction
tests/                    model smoke tests
TASKS.md                  build checklist
```

## Credits
Method: Afchar et al., WIFS 2018 ([arXiv:1809.00888](https://arxiv.org/abs/1809.00888)),
original code [DariusAf/MesoNet](https://github.com/DariusAf/MesoNet). This is an independent
educational reproduction.
