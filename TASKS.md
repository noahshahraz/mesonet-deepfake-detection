# TASKS — MesoNet reproduction

Work top-to-bottom. Check a box only when the task is done **and** verified. Scaffolded items
are already checked; the build phase starts at T5.

## Setup
- [x] T1 — Repo structure, `requirements.txt`, `.gitignore` (heavy folders ignored)
- [x] T2 — `configs/default.yaml` (no hardcoded hyperparameters)
- [x] T3 — Utils: `get_device` (MPS-first), `set_seed`, `load_config`
- [x] T4 — Models: `Meso4` and `MesoInception4` (+ `tests/test_models.py` passing)

## Data
- [x] T5 — `build_transforms` (resize 256, augmentation from cfg, normalize)
- [x] T6 — `build_dataloaders` (ImageFolder, assert real=0/fake=1, subsample, workers)
- [x] T7 — Download + arrange OpenForensics into `data/openforensics/{train,val,test}/{real,fake}` (see `scripts/download_data.md`) — stored at `~/mesonet-data/openforensics` (outside Drive); 140,002 / 39,428 / 10,905 images

## Training & evaluation
- [x] T8 — Wire model + data in `src/train.py`; move to `device`
- [x] T9 — Loss (`BCEWithLogitsLoss`) + `Adam(lr)`; epoch loop, checkpoint best val AUC, early stopping, logging
- [x] T10 — `src/eval.py`: load checkpoint, run test set, print metrics table
- [x] T11 — Metrics + plots: accuracy, AUC, precision/recall/F1, confusion matrix, ROC curve saved to `outputs/`
- [x] T12 — Train Meso-4 and MesoInception-4 on OpenForensics; record baseline numbers — test acc 0.871/0.871, AUC 0.953/0.952 (README)

## Paper reproduction (FaceForensics++)
- [x] T13 — Request FF++ access (Google form) — access granted; c23 videos (original/Deepfakes/Face2Face, 3×1000) at `~/mesonet-data/faceforensics_raw`
- [x] T14 — `scripts/extract_faces_ffpp.py`: frames → face crops → standard layout, split by source video — official splits; 59,997/60,000 frames detected; roots at `~/mesonet-data/ff_{deepfakes,face2face}`
- [x] T15 — Train per method (Deepfakes, Face2Face); target ~98% / ~95% accuracy, AUC ~0.99 — c23 results: Meso-4 DF 0.934/0.985, F2F 0.915/0.968; MesoInception-4 DF 0.910/0.982, F2F 0.923/0.970 (acc/AUC)
- [ ] T16 — Fill the "paper vs. mine" results table in README (FF++ only)

## Verification
- [ ] T17 — Verify implementation against the paper; write `docs/paper_diff.md` listing every
      difference (architecture, optimizer, augmentation, loss/sigmoid, thresholds)

## Phase 3 — Extensions
- [ ] T18 — Cross-dataset generalization: evaluate the SAME FF++-trained checkpoint on
      OpenForensics and 140k; build the generalization comparison table
- [ ] T19 — XceptionNet baseline: train/eval on the same splits; add to comparison
- [ ] T20 — Threshold tuning: sweep decision threshold (`utils.metrics.best_threshold`),
      report accuracy/F1 before vs. after per dataset
- [ ] T21 — Final README pass: results tables + one-line reproduce command + short write-up of the
      generalization finding
