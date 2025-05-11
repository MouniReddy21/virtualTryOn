#!/usr/bin/env python3
"""
Background removal utility using rembg.

Defines:
    remove_background(img_path: str) -> np.ndarray

Also provides a CLI for batch-processing a directory of images.
"""
import os
import argparse
from PIL import Image
import numpy as np
from rembg import remove


def remove_background(img_path: str) -> np.ndarray:
    """
    Remove the background from an image file.

    Args:
        img_path: Path to the input image (RGB or RGBA).

    Returns:
        A numpy array of shape (H, W, 4) in RGBA format, where the background
        has been made transparent.
    """
    # Load image and ensure RGBA mode
    with Image.open(img_path) as img:
        img = img.convert("RGBA")
        # rembg.remove accepts a PIL Image and returns a PIL Image
        result = remove(img)

    # Convert result to numpy RGBA array
    rgba_array = np.array(result)
    return rgba_array


def batch_remove(input_dir: str, output_dir: str) -> None:
    """
    Walks through input_dir, applies background removal to each image,
    and writes the RGBA PNG to output_dir preserving subfolders.
    """
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for root, _, files in os.walk(input_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue
            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                rgba = remove_background(in_path)
                # Save as PNG to preserve alpha
                Image.fromarray(rgba).save(out_path)
                print(f"Saved transparent: {out_path}")
            except Exception as e:
                print(f"Failed on {in_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove backgrounds from all images in a directory."
    )
    parser.add_argument(
        '--input-dir', '-i', required=True,
        help='Directory of input images.'
    )
    parser.add_argument(
        '--output-dir', '-o', required=True,
        help='Directory to save RGBA outputs.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_remove(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()