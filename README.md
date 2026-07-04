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
FaceForensics++, **c23 (HQ) compression**, per-image scoring on 20 frames/video, threshold 0.5
(untuned), official disjoint identity splits. The paper's ~98%/~95% come from the authors' own
lighter-compression dataset with per-video frame aggregation, so per-image c23 numbers landing a
few points lower is **expected, not a defect**.

| Model | Method | Paper acc.* | Mine acc. (c23) | Mine AUC |
|---|---|---|---|---|
| Meso-4 | Deepfakes | ~0.98 | 0.934 | 0.985 |
| MesoInception-4 | Deepfakes | ~0.98 | 0.910 | 0.982 |
| Meso-4 | Face2Face | ~0.95 | 0.915 | 0.968 |
| MesoInception-4 | Face2Face | ~0.95 | 0.923 | 0.970 |

*Two consistency checks hold: Deepfakes scores above Face2Face (the paper's difficulty ordering),
and AUCs of 0.97–0.985 sit right under the paper's ~0.99 — high AUC next to ~92% accuracy says
the 0.5 threshold is suboptimal, which threshold tuning (T20) addresses. Honestly noted: Meso-4
slightly outperforms MesoInception-4 on Deepfakes here (single-seed variance territory).

### Generalization — one FF++-trained model, evaluated across datasets
_Not paper-comparable; the point is how performance shifts off the training distribution._
Source model: `meso4_ff_deepfakes_best.pth` (FF++ Deepfakes, epoch 26), **weights frozen**,
threshold 0.5 (untuned). AUC is the primary cross-dataset metric — accuracy loss can be partly
threshold miscalibration, an AUC drop is true generalization loss.

| Eval dataset | Meso-4 acc. | Meso-4 AUC | Xception acc. |
|---|---|---|---|
| FF++ Deepfakes (in-domain reference) | 0.934 | 0.985 | _TBD (T19)_ |
| FF++ Face2Face (cross-method control) | 0.543 | 0.621 | — |
| OpenForensics | 0.467 | 0.405 | _TBD (T19)_ |
| 140k (StyleGAN) | 0.498 | 0.403 | _TBD (T19)_ |

**The finding is starker than a "drop":** cross-*method* transfer inside the same preprocessing
domain retains real signal (AUC 0.62), but cross-*dataset* transfer collapses to **below chance**
(AUC ≈ 0.40 on both external sets — a symmetric effect; the reverse direction,
OpenForensics→FF++, gives AUC 0.46). Below-chance AUC means the ranking *inverts*: the
mesoscopic compression/resampling cues MesoNet learns on c23 video frames anti-correlate with
fakeness elsewhere — on 140k the model calls nearly everything real (6/10,000 fakes flagged),
consistent with StyleGAN fakes looking *smoother* to a compression-artifact detector than FFHQ
reals. Threshold tuning cannot fix an inverted ranking; this is a genuine domain-shift result,
not a calibration artifact.

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
