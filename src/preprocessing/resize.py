#!/usr/bin/env python3
"""
Batch image resizer: walks a source directory, resizes all images, and saves them to a destination directory,
preserving subdirectory structure.
"""
import os
import argparse
from PIL import Image

# Supported image extensions
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def resize_image_file(
    input_path: str, output_path: str, size: tuple[int, int], resample: int
) -> None:
    """
    Resize a single image and save it.
    """
    # Open image and ensure RGB
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        resized = img.resize(size, resample)

    # Create output folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save in the same format as input
    resized.save(output_path)
    print(f"Resized: {input_path} -> {output_path}")


def batch_resize(
    src_dir: str, dst_dir: str, size: tuple[int, int], resample: int
) -> None:
    """
    Walk src_dir, resize all images, and save to dst_dir preserving relative paths.
    """
    for root, _, files in os.walk(src_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(in_path, src_dir)
                out_path = os.path.join(dst_dir, rel_path)
                resize_image_file(in_path, out_path, size, resample)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch resize images from a source directory into a destination directory."
    )
    parser.add_argument(
        "--src-dir", required=True, help="Path to input directory containing images."
    )
    parser.add_argument(
        "--dst-dir", required=True, help="Path to output directory for resized images."
    )
    parser.add_argument(
        "--width", type=int, required=True, help="Target width in pixels."
    )
    parser.add_argument(
        "--height", type=int, required=True, help="Target height in pixels."
    )
    parser.add_argument(
        "--resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bilinear",
        help="Resampling filter.",
    )
    return parser.parse_args()


def get_resample_filter(name: str) -> int:
    """
    Map resample name to PIL constant.
    """
    return {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }[name]


def main():
    args = parse_args()
    size = (args.width, args.height)
    resample = get_resample_filter(args.resample)

    batch_resize(args.src_dir, args.dst_dir, size, resample)


if __name__ == "__main__":
    main()
