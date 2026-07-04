# Getting the datasets

All datasets normalise to: `data/<name>/<split>/<real|fake>/*.jpg`

> **This machine:** datasets live at `~/mesonet-data/<name>/...` (outside the Google-Drive-synced
> repo) — pass `--data-root ~/mesonet-data/<name>` to `src.train` / `src.eval`, or set `data.root`.

## 1. OpenForensics — `data/openforensics/` (START HERE, no form)
Kaggle: `manjilkarki/deepfake-and-real-images` — 190,335 pre-cropped 256×256 faces
(train 140,002 / val 39,428 / test 10,905), already in real/fake folders.

```bash
# put kaggle.json in ~/.kaggle/ first (Kaggle > Settings > API > Create New Token)
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p data/ --unzip
# then arrange/rename into data/openforensics/{train,val,test}/{real,fake}/
```

## 2. 140k Real and Fake Faces — `data/faces140k/` (generalization test, no form)
Kaggle: `xhlulu/140k-real-and-fake-faces` — 140,002 images, 4.04 GB, StyleGAN fakes.
NOTE: fakes are fully-synthetic faces, NOT face-swaps → hardest generalization case.

```bash
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/ --unzip
```

## 3. FaceForensics++ — `data/faceforensics/` (PAPER-COMPARABLE, needs access)
Official (recommended): request access → https://github.com/ondyari/FaceForensics
Fill the Google form; you receive `download.py`. Pull the **c23** (HQ) compressed set for
`Deepfakes`, `Face2Face`, and `original` only (keeps it laptop-sized):

```bash
python download.py data/faceforensics_raw -d Deepfakes   -c c23 -t videos
python download.py data/faceforensics_raw -d Face2Face    -c c23 -t videos
python download.py data/faceforensics_raw -d original     -c c23 -t videos
# then extract + crop faces into the standard layout:
python scripts/extract_faces_ffpp.py --src data/faceforensics_raw --dst data/faceforensics \
       --frames-per-video 20
```

Kaggle mirrors (if the form is slow): `xdxd003/ff-c23` (videos),
`adham7elmy/faceforencispp-extracted-frames` (frames). Verify licensing/terms before use.
```
