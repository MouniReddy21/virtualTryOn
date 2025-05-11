#!/usr/bin/env python3
"""
Pose estimation utility supporting OpenPose or MediaPipe backends.

Usage (OpenPose):
  python src/preprocessing/detect_pose.py \
    --backend openpose \
    --openpose-bin /path/to/openpose/build/examples/openpose/openpose.bin \
    --input-dir inputs/resize/image \
    --write-json outputs/pose-json \
    --write-images outputs/pose-img

Usage (MediaPipe):
  python src/preprocessing/detect_pose.py \
    --backend mediapipe \
    --input-dir inputs/resize/image \
    --write-json outputs/pose-json \
    --write-images outputs/pose-img
"""
import os
import argparse
import subprocess
import json
from glob import glob

# Common libs
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Optional import for MediaPipe
try:
    import mediapipe as mp
except ImportError:
    mp = None


def run_openpose(input_dir, json_dir, img_dir, openpose_bin):
    """
    Run the OpenPose binary on a directory of images.
    """
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    cmd = [
        openpose_bin,
        "--image_dir",
        input_dir,
        "--write_json",
        json_dir,
        "--write_images",
        img_dir,
        "--display",
        "0",
        "--render_pose",
        "1",
        "--disable_blending",
        "true",
        "--hand",
    ]
    subprocess.run(cmd, check=True)


def run_mediapipe(input_dir, json_dir, img_dir):
    """
    Run MediaPipe Pose on each image and save keypoints in OpenPose-like JSON.
    """
    if mp is None:
        raise ImportError("MediaPipe not installed. pip install mediapipe")

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS

    with mp_pose.Pose(static_image_mode=True) as pose:
        for img_path in sorted(glob(os.path.join(input_dir, "*"))):
            base = os.path.basename(img_path)
            name, _ = os.path.splitext(base)

            # Read & process
            img_bgr = cv2.imread(img_path)
            h, w = img_bgr.shape[:2]
            results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            # Build keypoints list: [x, y, visibility] * 33 landmarks
            keypoints = []
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x * w, lm.y * h, lm.visibility])
            else:
                keypoints = [0.0] * (33 * 3)

            # Save JSON
            out_json = os.path.join(json_dir, f"{name}_keypoints.json")
            with open(out_json, "w") as f:
                json.dump({"people": [{"pose_keypoints_2d": keypoints}]}, f)

            # Draw skeleton overlay
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_rgb)
            if results.pose_landmarks:
                for i, j in connections:
                    p1 = results.pose_landmarks.landmark[i]
                    p2 = results.pose_landmarks.landmark[j]
                    draw.line(
                        [(p1.x * w, p1.y * h), (p2.x * w, p2.y * h)],
                        fill="white",
                        width=2,
                    )
            out_img = os.path.join(img_dir, f"{name}_pose.png")
            img_rgb.save(out_img)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose estimation: OpenPose or MediaPipe"
    )
    parser.add_argument("--backend", choices=["openpose", "mediapipe"], required=True)
    parser.add_argument("--input-dir", "-i", required=True)
    parser.add_argument("--write-json", "-j", required=True)
    parser.add_argument("--write-images", "-v", required=True)
    parser.add_argument("--openpose-bin", help="Path to OpenPose binary")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backend == "openpose":
        if not args.openpose_bin:
            raise ValueError("Must provide --openpose-bin for OpenPose backend")
        run_openpose(
            args.input_dir, args.write_json, args.write_images, args.openpose_bin
        )
    else:
        run_mediapipe(args.input_dir, args.write_json, args.write_images)


if __name__ == "__main__":
    main()
