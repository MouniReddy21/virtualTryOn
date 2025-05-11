#!/usr/bin/env python3
"""
Generate binary cloth masks for every image in a directory using U²-Net.

Usage:
  python src/preprocessing/cloth_mask.py \
    --input-dir datasets/street_tryon/train/cloth \
    --output-dir preprocessed/mask \
    --checkpoint checkpoints/cloth_segm_u2net_latest.pth \
    [--width 768 --height 768]
"""
import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

# Import the U2NET model; adjust this path if your project structure differs
from src.models.u2net.model.u2net import U2NET


def load_model(checkpoint_path: str, device: torch.device) -> U2NET:
    """
    Load U²-Net with given checkpoint onto the specified device.
    """
    net = U2NET(in_ch=3, out_ch=4)
    state = torch.load(checkpoint_path, map_location=device)
    # Strip module prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def generate_mask(
    net: U2NET, img_path: str, transform: transforms.Compose, device: torch.device
) -> Image.Image:
    """
    Run a single image through U²-Net and return a binary PIL mask.
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = net(tensor)[0]  # [4, H, W]
        prob = F.softmax(pred, dim=0)  # class probabilities
        cls = torch.argmax(prob, dim=0)  # [H, W]

    mask = (cls.cpu().numpy() != 0).astype(np.uint8) * 255
    return Image.fromarray(mask)


def batch_process(
    input_dir: str,
    output_dir: str,
    net: U2NET,
    transform: transforms.Compose,
    device: torch.device,
) -> None:
    """
    Walk input_dir, generate and save masks to output_dir, preserving subfolders.
    """
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue
            in_path = os.path.join(root, fname)
            rel = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            mask_img = generate_mask(net, in_path, transform, device)
            mask_img.save(out_path)
            print(f"Saved mask: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate cloth masks with U²-Net"
    )
    parser.add_argument("--input-dir", required=True, help="Directory of cloth images")
    parser.add_argument(
        "--output-dir", required=True, help="Where to save binary masks"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/cloth_segm_u2net_latest.pth",
        help="Path to U²-Net .pth checkpoint",
    )
    parser.add_argument("--width", type=int, default=768, help="Resize width")
    parser.add_argument("--height", type=int, default=768, help="Resize height")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading U²-Net from {args.checkpoint} on {device}")
    net = load_model(args.checkpoint, device)

    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    batch_process(args.input_dir, args.output_dir, net, transform, device)


if __name__ == "__main__":
    main()
