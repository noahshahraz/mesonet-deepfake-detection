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
Arrange a dataset per [`scripts/download_data.md`](scripts/download_data.md) first; pass
`--data-root` if it lives outside the repo (e.g. `~/mesonet-data/openforensics` on this machine —
omit the flag if you used the default `data/openforensics`).

```bash
# train then evaluate Meso-4 end-to-end (full OpenForensics: ~2.5 h on an M-series MacBook, MPS)
python -m src.train --config configs/default.yaml --data-root ~/mesonet-data/openforensics && \
python -m src.eval  --config configs/default.yaml --data-root ~/mesonet-data/openforensics --checkpoint checkpoints/best.pth
```

Reviewer-friendly fast variant (~5 min, same code path, lower numbers). Caution: run names are
`<model>_<dataset>`, so this **overwrites** `checkpoints/best.pth` and the full-run
`meso4_openforensics` checkpoint/outputs if you have them:
```bash
python -m src.train --config configs/default.yaml --data-root ~/mesonet-data/openforensics --max-per-class-train 2000 --epochs 3 && \
python -m src.eval  --config configs/default.yaml --data-root ~/mesonet-data/openforensics --checkpoint checkpoints/best.pth
```

## Results

### Baseline — OpenForensics (NOT paper-comparable)
Full 190k-image Kaggle set (`manjilkarki/deepfake-and-real-images`), full train split (140,002
images), best-val-AUC checkpoint, threshold 0.5. Both models reach val AUC ≈ 0.990; the test
split is markedly harder than val (val acc ~0.93 vs test ~0.87) — a known distribution quirk of
this dataset, reported as-is.

| Model | Test acc. | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Meso-4 | 0.867 | 0.946 | 0.881 | 0.852 | 0.866 |
| MesoInception-4 | 0.871 | 0.952 | 0.882 | 0.860 | 0.871 |

_Run-to-run note: a same-seed Meso-4 retrain (MPS training is not bit-deterministic) moved test
accuracy 0.871→0.867 and AUC 0.953→0.946 at identical val AUC (0.990) — a fair picture of
single-run variance on this test split; the table reports the run whose artifacts ship in
`outputs/`._

### Reproduction — paper vs. this repo (FaceForensics++)
FaceForensics++, **c23 (HQ) compression**, per-image scoring on 20 frames/video, threshold 0.5
(untuned), official disjoint identity splits. The paper's ~98%/~95% come from the authors' own
lighter-compression dataset with per-video frame aggregation, so per-image c23 numbers landing a
few points lower is **expected, not a defect**.

| Model | Method | Paper acc.* | Mine acc. (c23) | Mine acc. (tuned t†) | Mine AUC |
|---|---|---|---|---|---|
| Meso-4 | Deepfakes | ~0.98 | 0.934 | 0.934 (t=0.50) | 0.985 |
| MesoInception-4 | Deepfakes | ~0.98 | 0.910 | 0.928 (t=0.69) | 0.982 |
| Meso-4 | Face2Face | ~0.95 | 0.915 | 0.913 (t=0.53) | 0.968 |
| MesoInception-4 | Face2Face | ~0.95 | 0.923 | 0.922 (t=0.66) | 0.970 |

*Two consistency checks hold: Deepfakes scores above Face2Face (the paper's difficulty ordering),
and AUCs of 0.97–0.985 sit right under the paper's ~0.99. Honestly noted: Meso-4 slightly
outperforms MesoInception-4 on Deepfakes here (single-seed variance territory). For a like-for-like
anchor, the paper's own *per-image* c23 Face2Face accuracies are 92.4%/93.4% — within ~1 pt of ours;
the 95–98% headlines are per-video aggregates.

†Threshold selected on the **validation** split (`scripts/tune_threshold.py`, T20), never on test.
Tuning only matters where precision/recall was imbalanced — MesoInception-4/Deepfakes gains
+1.8 pts at t=0.69; the other three runs were already calibrated at 0.5 (the tiny negative deltas
are honest val→test disagreement). The residual gap to the paper's headlines is therefore the
per-image c23 protocol, not calibration.

### Generalization — one FF++-trained model, evaluated across datasets
_Not paper-comparable; the point is how performance shifts off the training distribution._
Source model: `meso4_ff_deepfakes_best.pth` (FF++ Deepfakes, epoch 26), **weights frozen**,
threshold 0.5 (untuned). AUC is the primary cross-dataset metric — accuracy loss can be partly
threshold miscalibration, an AUC drop is true generalization loss.

| Eval dataset | Meso-4 acc. | Meso-4 AUC | Xception acc. | Xception AUC |
|---|---|---|---|---|
| FF++ Deepfakes (in-domain reference) | 0.934 | 0.985 | 0.976 | 0.997 |
| FF++ Face2Face (cross-method control) | 0.543 | 0.621 | 0.519 | 0.718 |
| OpenForensics | 0.467 | 0.405 | 0.477 | 0.295 |
| 140k (StyleGAN) | 0.498 | 0.403 | 0.499 | 0.334 |

**Generalization findings.** Neither model transfers across datasets. Cross-*method* transfer
inside the same preprocessing domain retains usable signal (Meso-4 AUC 0.62; Xception 0.72), but
cross-*dataset* the signal is essentially lost: Meso-4's AUC ≈ 0.40 on both external sets is a
**mild ranking inversion** — a faint anti-correlation, not a reliable inverted classifier — and
transfer fails in the reverse direction too (OpenForensics→FF++ sits at chance: AUC 0.46–0.50
across two same-seed training runs). The
mechanism is consistent with low-level cue mismatch: the mesoscopic compression/resampling
artifacts learned on c23 video frames point the wrong way elsewhere — on 140k the model flags
6/10,000 fakes, plausibly because StyleGAN fakes look *smoother* to a compression-artifact
detector than FFHQ reals. Xception (`legacy_xception`, timm, ImageNet-pretrained, **20.8M params
vs MesoNet's 28k**, fine-tuned on the same root at 256×256 — global pooling makes the native-299
input a non-issue) answers the capacity question: clearly stronger in-domain (0.976/0.997) and
across methods, yet its cross-dataset inversion is *more* pronounced (AUC 0.29/0.33). The
brittleness comes from single-domain training, not model capacity. Note that a decision
threshold cannot rescue an AUC below 0.5 (the ranking itself is inverted), so the cross-dataset
numbers stand as-is — threshold tuning (T20) applies only in-domain.

## Limitations / future work
- **Single seed** (42) throughout — the Meso-4 vs MesoInception-4 ordering on Deepfakes sits
  within plausible seed variance; multi-seed runs would tighten every table.
- **Per-image evaluation only.** The paper's 98%/95% headlines average predictions per video;
  adding per-video aggregation is the most likely single step toward closing the residual gap.
- **Loss deviation:** the paper trains squared error on a sigmoid; we use `BCEWithLogitsLoss`
  (see [`docs/paper_diff.md`](docs/paper_diff.md) for this and every other difference, incl. the
  paper's step lr schedule and hue augmentation that we omit).

## Repo layout
```
configs/default.yaml      all hyperparameters (nothing hardcoded)
src/models/               Meso4, MesoInception4 (+ timm Xception baseline)
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
