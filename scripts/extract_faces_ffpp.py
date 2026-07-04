"""Extract frames from FaceForensics++ videos and crop faces into the standard layout.

    data/faceforensics/<split>/<real|fake>/*.jpg

STATUS: SKELETON — implement in the build phase (TASKS.md T15).

Plan:
  - Walk src for original/ (-> real) and Deepfakes/, Face2Face/ (-> fake) c23 videos.
  - Sample N frames per video (cfg --frames-per-video).
  - Detect + crop the face per frame (facenet-pytorch MTCNN, or OpenCV Haar as a light
    fallback). Resize to 256x256.
  - Split by SOURCE VIDEO id (not by frame) into train/val/test to avoid leakage.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="raw FF++ download dir")
    p.add_argument("--dst", required=True, help="output dir (standard layout)")
    p.add_argument("--frames-per-video", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError("Implement in build phase — see TASKS.md T15")


if __name__ == "__main__":
    main()
