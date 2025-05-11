#!/usr/bin/env python3
"""
runner.py: orchestrates the full preprocessing pipeline for a cloth-person pair.

Steps:
 1. Copy & resize cloth & person images
 2. Generate cloth mask via U²-Net
 3. Remove background from person via rembg
 4. Parse human segmentation via SCHP
 5. Estimate pose via OpenPose or MediaPipe

Outputs all artifacts under a single --out-dir, preserving the structure:
    out-dir/
      ├── cloth/
      ├── image/
      ├── cloth-mask/
      ├── image-parse/
      ├── openpose-img/
      └── openpose-json/
"""
import os
import argparse
import shutil
from uuid import uuid4
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Relative imports from preprocessing modules
from .resize import resize_image_file
from .cloth_mask import load_model as load_mask_model, generate_mask
from .remove_bg import remove_background
from .parse_human import parse_human
from .detect_pose import run_openpose, run_mediapipe


def get_resample_filter(name: str):
    return {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }[name]


def run_all(
    person_path: str,
    cloth_path: str,
    out_dir: str,
    width: int,
    height: int,
    resample: str,
    mask_ckpt: str,
    parse_ckpt: str,
    parse_dataset: str,
    pose_backend: str,
    openpose_bin: str | None
) -> None:
    # 1. Prepare output subdirectories
    cloth_dir        = os.path.join(out_dir, 'cloth')
    image_dir        = os.path.join(out_dir, 'image')
    cloth_mask_dir   = os.path.join(out_dir, 'cloth-mask')
    image_parse_dir  = os.path.join(out_dir, 'image-parse')
    openpose_img_dir = os.path.join(out_dir, 'openpose-img')
    openpose_json_dir= os.path.join(out_dir, 'openpose-json')

    for d in [cloth_dir, image_dir, cloth_mask_dir, image_parse_dir, openpose_img_dir, openpose_json_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy originals
    cloth_fname = os.path.basename(cloth_path)
    image_fname = os.path.basename(person_path)
    shutil.copy2(cloth_path, os.path.join(cloth_dir, cloth_fname))
    shutil.copy2(person_path, os.path.join(image_dir, image_fname))

    # 2. Resize cloth & person
    resample_filter = get_resample_filter(resample)
    resize_image_file(
        os.path.join(cloth_dir, cloth_fname),
        os.path.join(cloth_dir, cloth_fname),
        (width, height), resample_filter
    )
    resize_image_file(
        os.path.join(image_dir, image_fname),
        os.path.join(image_dir, image_fname),
        (width, height), resample_filter
    )

    # 3. Generate cloth mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
    mask_net = load_mask_model(mask_ckpt, device)
    mask_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    mask = generate_mask(
        mask_net,
        os.path.join(cloth_dir, cloth_fname),
        mask_transform,
        device
    )
    mask.save(os.path.join(cloth_mask_dir, os.path.splitext(cloth_fname)[0] + '.png'))
    print("Cloth mask saved.")

    # 4. Remove background from person
    print("Removing background from person image...")
    rgba = remove_background(os.path.join(image_dir, image_fname))
    Image.fromarray(rgba).save(os.path.join(image_dir, image_fname))

    # 5. Human parsing
    print("Parsing human segmentation...")
    img_arr = np.array(Image.fromarray(rgba).convert('RGB'), dtype=np.uint8)
    parse_map = parse_human(img_arr, parse_ckpt, parse_dataset)
    Image.fromarray(parse_map).save(os.path.join(image_parse_dir, os.path.splitext(image_fname)[0] + '.png'))

    # 6. Pose estimation
    print(f"Estimating pose via {pose_backend}...")
    if pose_backend == 'openpose':
        if not openpose_bin:
            raise ValueError('`--openpose-bin` is required for the openpose backend')
        run_openpose(image_dir, openpose_json_dir, openpose_img_dir, openpose_bin)
    else:
        run_mediapipe(image_dir, openpose_json_dir, openpose_img_dir)

    print(f"Preprocessing complete. All artifacts in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Full preprocessing runner for virtual try-on')
    parser.add_argument('--person',        '-p', required=True, help='Path to person image')
    parser.add_argument('--cloth',         '-c', required=True, help='Path to cloth image')
    parser.add_argument('--out-dir',       '-o', required=True, help='Directory to emit outputs')
    parser.add_argument('--width',          type=int, default=768, help='Resize width')
    parser.add_argument('--height',         type=int, default=1024, help='Resize height')
    parser.add_argument('--resample',       choices=['nearest','bilinear','bicubic','lanczos'], default='bilinear')
    parser.add_argument('--mask-ckpt',     default='checkpoints/cloth_segm_u2net_latest.pth')
    parser.add_argument('--parse-ckpt',    default='checkpoints/final.pth')
    parser.add_argument('--parse-dataset', choices=['lip','atr','pascal'], default='lip')
    parser.add_argument('--pose-backend',   choices=['openpose','mediapipe'], default='mediapipe')
    parser.add_argument('--openpose-bin',   help='Path to OpenPose binary (if using openpose)')
    args = parser.parse_args()
    run_all(
        args.person,
        args.cloth,
        args.out_dir,
        args.width,
        args.height,
        args.resample,
        args.mask_ckpt,
        args.parse_ckpt,
        args.parse_dataset,
        args.pose_backend,
        args.openpose_bin
    )


if __name__ == '__main__':
    main()
