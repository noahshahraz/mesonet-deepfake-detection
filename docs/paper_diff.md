# Paper vs. implementation — differences (T17)

Verified against a full re-read of MesoNet (Afchar et al., WIFS 2018,
[arXiv:1809.00888](https://arxiv.org/abs/1809.00888)) on 2026-07-04. Every point where this repo
differs from the described method, with a one-line justification. Param counts and architecture
were cross-checked by exact parameter-count match (see below).

## Deviations (confirmed, kept deliberately)

| Item | Paper | This repo | Why |
|---|---|---|---|
| **Loss** | **Squared error** ½(a−y)² on sigmoid output (§ III; the scaffold's assumption of binary cross-entropy was itself wrong) | Single logit + `BCEWithLogitsLoss` | BCE is the canonical binary-classification loss; numerically stable in logit form. Same argmax behaviour; not identical gradients to MSE — a knowing deviation. |
| Output layer | Dense(1) + sigmoid | Dense(1) raw logit (sigmoid applied only at eval) | Equivalent probabilities; avoids saturated-sigmoid gradient issues |
| **LR schedule** | 10⁻³, ÷10 every 1000 iterations down to 10⁻⁶ | Constant 10⁻³ (Adam), early stopping on val AUC | Simpler; converged fine (val AUC 0.98+). A schedule could add a fraction of a point. |
| Batch size | 75 | 76 (`train.batch_size`) | Effectively identical; even number packs better |
| Early stopping | Not specified | Patience 8 on val AUC, max 50 epochs | Paper gives iterations, not epochs; we stop on plateau |
| Input normalization | Keras code rescales to [0,1] | ImageNet mean/std (`data.normalize: true`) | Standard PyTorch practice; BN largely absorbs the affine difference. Config supports `normalize: null` for [0,1]. |
| Augmentation | Zoom, rotation, horizontal flip, brightness **and hue** changes (magnitudes unspecified) | Flip, rotation 15°, zoom ±0.1, brightness ±0.1 — **no hue jitter** | Magnitudes are unspecified in the paper; hue omitted (face hue is a plausible forgery cue we chose not to perturb) |
| Face extraction | Viola-Jones detector + NN landmark **alignment**, ~50 faces/scene, **manually reviewed** | MTCNN largest-face, margin 0.3, square crop, 20 frames/video, no alignment, no manual review | MTCNN is stronger than Viola-Jones (3/60,000 misses); no alignment step — BN + augmentation compensate; manual review impractical for a reproduction |
| Training data | Their own Deepfake set (5,111 forged / 7,250 real crops) + FF++ Face2Face (300 videos, 4,500/4,500) | FF++ c23 only, official 720/140/140 identity splits → ~14.4k/class train per method | Authors' dataset is not distributed; FF++ is the reproducible benchmark |
| Evaluation | Per-image **and** per-video (prediction averaged over video); headline 98.4%/95.3% are **per-video** | Both, since hardening task 3: per-image plus per-video via `scripts/eval_per_video.py` (mean P(fake) over 20 frames/video) | **Resolved.** Per-video Face2Face = 0.950–0.951 vs the paper's 95.3% (match, same c23 compression, 3 seeds); per-video Deepfakes = 0.950–0.955 vs their 98.4% on the authors' unreleased lighter-compression set |
| Framework | Keras/TensorFlow | PyTorch 2.12 | See framework-defaults notes below |
| Device | GPU (CUDA implied) | Apple **MPS** (`src.utils.get_device`) | Target hardware |

### Framework-default differences (Keras → PyTorch)
- **BatchNorm momentum:** Keras default decay 0.99 vs PyTorch `momentum=0.1` (≙ decay 0.9) —
  running stats adapt faster here; negligible at our epoch counts.
- **BatchNorm eps:** Keras 1e-3 vs PyTorch 1e-5.
- **Adam eps:** Keras 1e-7 vs PyTorch 1e-8; betas match the paper's (0.9, 0.999) in both.
- **Padding:** Keras `'same'` vs our `padding=kernel//2` — identical for odd kernels at stride 1;
  pooling divides 256 exactly at every stage, so no edge-padding asymmetry anywhere.

## Architecture checklist — verified against the paper
- [x] Meso-4: 4 conv blocks, filters 8 / 8 / 16 / 16
- [x] Kernel sizes: 3×3 (block 1), 5×5 (blocks 2–4)
- [x] Pooling: 2×2 after blocks 1–3, **4×4** after block 4
- [x] Block order: Conv → ReLU → BatchNorm → MaxPool (conv activations ReLU, per paper)
- [x] FC head: Dropout(0.5) → Dense(16) → LeakyReLU(0.1) → Dropout(0.5) → Dense(1)
- [x] MesoInception-4: first two blocks are Inception modules
- [x] Inception branches: 1×1 conv + three 1×1→3×3 branches with **dilation 1 / 2 / 3**, concatenated
- [x] Inception channel widths: (1,4,4,2) then (2,4,4,2) — confirmed via exact param-count match
- [x] Parameter counts: **27,977 (Meso-4) and 28,615 (MesoInception-4) — exact match to the paper**

## Training setup
- [x] Optimizer: Adam, lr 1e-3, betas (0.9, 0.999) — paper-matching (lr *schedule* differs, see table)
- [x] Loss: **differs** — paper uses squared error; we use BCE-with-logits (see table)
- [x] Input: 256×256×3 face crops (paper-matching)
- [x] Augmentation: compared above — hue jitter omitted, magnitudes are our choices
- [x] Decision threshold: 0.5 (paper-implied) — Phase 3 T20 tunes it on saved probs
- [x] Training set size: paper ~12.4k (Deepfake) / 9k (F2F) images; ours ~28.8k per method
  (720 train videos × 20 frames × 2 classes)

## Evaluation / results
- [x] Metrics: per-image accuracy (+ AUC/P/R/F1, which the paper does not report) AND, since
  hardening task 3, per-video aggregation matching the paper's protocol — Face2Face lands on the
  paper's 95.3% within noise (0.950–0.951, 3 seeds)
- [x] Reproduced FF++ c23 per-image accuracy: Meso-4 93.4% (DF) / 91.5% (F2F);
  MesoInception-4 91.0% (DF) / 92.3% (F2F). Paper per-image c23 Face2Face: 92.4% / 93.4%;
  paper per-video headlines: 98.4% (DF, their dataset) / 95.3% (F2F c23).

## Open questions / anything that could not be matched
- Exact augmentation magnitudes and the hue-jitter range (paper does not specify).
- The paper's Deepfake results use the authors' own unreleased dataset; only Face2Face numbers
  are directly FF++-comparable at matching compression.
- Frame-selection heuristic ("proportional to camera-angle/illumination changes, ~50/scene") is
  not reproducible exactly; we sample 20 evenly spaced frames.
