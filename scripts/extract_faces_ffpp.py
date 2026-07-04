"""Extract frames from FaceForensics++ videos and crop faces into the standard layout.

Produces one dataset root per manipulation method, each holding original frames as `real`
and that method's frames as `fake`:

    <dst-root>/ff_deepfakes/<split>/{real,fake}/*.jpg      (original vs Deepfakes)
    <dst-root>/ff_face2face/<split>/{real,fake}/*.jpg      (original vs Face2Face)

Splits follow the OFFICIAL FF++ split files (scripts/ff_splits/{train,val,test}.json,
from github.com/ondyari/FaceForensics), so source identities never cross splits.
Original videos are decoded/detected once and written to every method root.

Face detection: MTCNN (facenet-pytorch), largest face, box expanded by --margin and
squared before the 256x256 resize. OpenCV Haar is a per-frame fallback; frames where
both detectors fail are skipped (counted in the summary).

Usage:
    python scripts/extract_faces_ffpp.py --src ~/mesonet-data/faceforensics_raw \
        --dst-root ~/mesonet-data --frames-per-video 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

METHODS = {
    "Deepfakes": ("ff_deepfakes", "manipulated_sequences/Deepfakes/c23/videos"),
    "Face2Face": ("ff_face2face", "manipulated_sequences/Face2Face/c23/videos"),
}
ORIGINAL_SUBDIR = "original_sequences/youtube/c23/videos"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(Path.home() / "mesonet-data/faceforensics_raw"),
                   help="raw FF++ download dir")
    p.add_argument("--dst-root", default=str(Path.home() / "mesonet-data"),
                   help="parent dir for the per-method dataset roots")
    p.add_argument("--splits-dir", default=str(Path(__file__).parent / "ff_splits"),
                   help="dir with the official train/val/test.json")
    p.add_argument("--frames-per-video", type=int, default=20)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--margin", type=float, default=0.3,
                   help="fractional box expansion before cropping")
    p.add_argument("--device", default=None, help="mtcnn device (default: src.utils.get_device)")
    return p.parse_args()


def load_split_pairs(splits_dir: str | Path) -> dict[str, list[list[str]]]:
    return {s: json.load(open(Path(splits_dir) / f"{s}.json")) for s in SPLITS}


def sample_frames(video_path: Path, n: int) -> list[tuple[int, np.ndarray]]:
    """Return up to n evenly spaced (frame_index, RGB frame) pairs."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    out = []
    for idx in sorted({int(i) for i in np.linspace(0, total - 1, n)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


class FaceCropper:
    """MTCNN largest-face detector with an OpenCV Haar fallback."""

    def __init__(self, device: torch.device, image_size: int, margin: float):
        self.mtcnn = MTCNN(select_largest=True, device=device)
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_size = image_size
        self.margin = margin

    def _detect(self, rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        boxes, _ = self.mtcnn.detect(Image.fromarray(rgb))
        if boxes is not None and len(boxes):
            return tuple(boxes[0])  # select_largest=True puts the largest first
        # Haar fallback
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                           minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (x, y, x + w, y + h)
        return None

    def crop(self, rgb: np.ndarray) -> Image.Image | None:
        box = self._detect(rgb)
        if box is None:
            return None
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        # expand by margin, then square up around the centre
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * (1 + self.margin)
        half = side / 2
        x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
        y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        face = Image.fromarray(rgb[y1:y2, x1:x2])
        return face.resize((self.image_size, self.image_size), Image.BILINEAR)


def extract_video(video_path: Path, out_dirs: list[Path], stem: str, cropper: FaceCropper,
                  n_frames: int, stats: Counter) -> None:
    """Crop faces from one video into every dir in out_dirs (skips if already extracted)."""
    if all(any(d.glob(f"{stem}_f*.jpg")) for d in out_dirs):
        stats["videos_skipped_existing"] += 1
        return
    if not video_path.is_file():
        stats["videos_missing"] += 1
        return
    frames = sample_frames(video_path, n_frames)
    if not frames:
        stats["videos_unreadable"] += 1
        return
    written = 0
    for idx, rgb in frames:
        face = cropper.crop(rgb)
        if face is None:
            stats["frames_no_face"] += 1
            continue
        for d in out_dirs:
            face.save(d / f"{stem}_f{idx:04d}.jpg", quality=95)
        written += 1
    stats["videos_done"] += 1
    stats["frames_written"] += written


def main() -> None:
    args = parse_args()
    # repo-local import so the script works when run as `python scripts/extract_faces_ffpp.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import get_device

    device = torch.device(args.device) if args.device else get_device()
    print(f"[extract] MTCNN device = {device}")
    cropper = FaceCropper(device, args.image_size, args.margin)

    src = Path(args.src).expanduser()
    dst_root = Path(args.dst_root).expanduser()
    split_pairs = load_split_pairs(args.splits_dir)

    # make all split/class dirs up front
    for method_root, _ in METHODS.values():
        for split in SPLITS:
            for cls in ("real", "fake"):
                (dst_root / method_root / split / cls).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for split, pairs in split_pairs.items():
        # originals -> real in EVERY method root (decode + detect once)
        ids = sorted({v for p in pairs for v in p})
        real_dirs = [dst_root / root / split / "real" for root, _ in METHODS.values()]
        for i, vid in enumerate(ids):
            extract_video(src / ORIGINAL_SUBDIR / f"{vid}.mp4", real_dirs, vid, cropper,
                          args.frames_per_video, stats)
            if (i + 1) % 50 == 0:
                print(f"[extract] {split}/real: {i + 1}/{len(ids)} videos")
        # manipulated -> fake in the matching method root (both pair orderings exist)
        names = ["_".join(p) for p in pairs] + ["_".join(p[::-1]) for p in pairs]
        for method, (root, subdir) in METHODS.items():
            fake_dir = [dst_root / root / split / "fake"]
            for i, name in enumerate(sorted(names)):
                extract_video(src / subdir / f"{name}.mp4", fake_dir, name, cropper,
                              args.frames_per_video, stats)
                if (i + 1) % 50 == 0:
                    print(f"[extract] {split}/fake[{method}]: {i + 1}/{len(names)} videos")
        print(f"[extract] finished split '{split}': {dict(stats)}")

    print("\n[extract] per-split image counts:")
    for method, (root, _) in METHODS.items():
        for split in SPLITS:
            counts = {cls: len(list((dst_root / root / split / cls).glob("*.jpg")))
                      for cls in ("real", "fake")}
            print(f"  {root}/{split}: real {counts['real']}, fake {counts['fake']}")
    print(f"[extract] totals: {dict(stats)}")


if __name__ == "__main__":
    main()
