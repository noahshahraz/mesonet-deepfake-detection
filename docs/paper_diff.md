# Paper vs. implementation — differences (T17)

Fill this in after the pipeline works. Re-read MesoNet (Afchar et al., WIFS 2018,
[arXiv:1809.00888](https://arxiv.org/abs/1809.00888)) and record every point where this repo
differs from the described method, plus a one-line justification. Goal: an honest, reviewer-ready
account of the reproduction.

## Known deviations (already in the scaffold — confirm and keep)
| Item | Paper | This repo | Why |
|---|---|---|---|
| Output layer | Dense(1) + **sigmoid**, binary cross-entropy | Dense(1) **logit** + `BCEWithLogitsLoss` | Numerically equivalent, more stable |
| Batch size | ~75 | 76 | Matches paper; net is tiny so memory is a non-issue |
| Framework | Keras / TensorFlow | PyTorch | Check BatchNorm momentum & padding defaults differ |
| Device | GPU (CUDA) | Apple **MPS** | Target hardware |

## Architecture checklist — confirm each matches (tick when verified against the paper)
- [ ] Meso-4: 4 conv blocks, filters 8 / 8 / 16 / 16
- [ ] Kernel sizes: 3×3 (block 1), 5×5 (blocks 2–4)
- [ ] Pooling: 2×2 after blocks 1–3, **4×4** after block 4
- [ ] Block order: Conv → ReLU → BatchNorm → MaxPool
- [ ] FC head: Dropout(0.5) → Dense(16) → LeakyReLU(0.1) → Dropout(0.5) → Dense(1)
- [ ] MesoInception-4: first two blocks are Inception modules
- [ ] Inception branches: 1×1 conv + three 1×1→3×3 branches with **dilation 1 / 2 / 3**, concatenated
- [ ] Inception channel widths: (1,4,4,2) then (2,4,4,2)
- [ ] Parameter counts ≈ 27,977 (Meso-4) and 28,615 (MesoInception-4)

## Training setup
- [ ] Optimizer: Adam, lr 1e-3 — confirm betas/eps if the paper specifies
- [ ] Loss: binary cross-entropy
- [ ] Input: 256×256×3, faces cropped/aligned
- [ ] Data augmentation: compare paper's set (zoom, rotation, brightness, flip) to `configs/default.yaml`
- [ ] Decision threshold: 0.5 (paper) — Phase 3 T20 tunes this
- [ ] Training set size: paper ~19k images — record the actual count used per dataset

## Evaluation / results
- [ ] Metric definitions match (per-image accuracy; note if paper aggregates per-video)
- [ ] Record reproduced FF++ numbers vs. paper here and in the README table

## Open questions / anything I could not match
- (list here)
