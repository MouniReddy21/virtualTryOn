# src/inference/run_single.py

import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from types import SimpleNamespace
from src.models.tryon_model import TryOnModel


def load_image(path, mode="RGB"):
    return Image.open(path).convert(mode)


def image_to_tensor(img, normalize=True):
    t = transforms.ToTensor()(img)
    if normalize:
        # maps [0,1] to [-1,1]
        t = (t - 0.5) * 2.0
    return t.unsqueeze(0)  # add batch dimension


def load_pose_render(path, size):
    # pose overlay is already visualized as an RGB image
    img = load_image(path, "RGB").resize(size, Image.BILINEAR)
    return transforms.ToTensor()(img).unsqueeze(0)  # [1,3,H,W] in [0,1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        required=True,
        help="output folder from runner.py (contains image/, cloth/, etc.)",
    )
    p.add_argument(
        "--checkpoints",
        required=True,
        help="folder containing seg_final.pth, gmm_final.pth, alias_final.pth",
    )
    p.add_argument("--out", required=True, help="where to save the final try-on PNG")
    p.add_argument("--width", type=int, default=768, help="full-resolution width")
    p.add_argument("--height", type=int, default=1024, help="full-resolution height")
    args = p.parse_args()

    inp = args.inputs
    W_full, H_full = args.width, args.height

    # ─── 1) Load full-resolution artifacts ─────────────────────────────────────────
    person_p = os.path.join(
        inp, "image", sorted(os.listdir(os.path.join(inp, "image")))[0]
    )
    cloth_p = os.path.join(
        inp, "cloth", sorted(os.listdir(os.path.join(inp, "cloth")))[0]
    )
    mask_p = os.path.join(
        inp, "cloth-mask", sorted(os.listdir(os.path.join(inp, "cloth-mask")))[0]
    )
    parse_dir = os.path.join(inp, "image-parse")
    if not os.path.isdir(parse_dir):
        raise FileNotFoundError(f"Expected directory not found: {parse_dir}")
    parse_files = [
        f for f in sorted(os.listdir(parse_dir)) if f.lower().endswith((".png", ".jpg"))
    ]
    if not parse_files:
        raise FileNotFoundError(
            f"No human‐parse maps in {parse_dir}. "
            "Did you run `python -m src.processing.runner` first?"
        )
    parse_p = os.path.join(parse_dir, parse_files[0])
    # parse_p = os.path.join(
    #     inp, "image-parse", sorted(os.listdir(os.path.join(inp, "image-parse")))[0]
    # )
    pose_p = os.path.join(
        inp, "openpose-img", sorted(os.listdir(os.path.join(inp, "openpose-img")))[0]
    )

    person_full = load_image(person_p).resize((W_full, H_full), Image.BILINEAR)
    cloth_full = load_image(cloth_p).resize((W_full, H_full), Image.BILINEAR)
    mask_full = load_image(mask_p, "L").resize((W_full, H_full), Image.NEAREST)
    # parse_full = load_image(parse_p, "L").resize((W_full, H_full), Image.NEAREST)
    # Open the SCHP PNG _with_ its palette indices intact:
    # parse_full = Image.open(parse_p).resize((W_full, H_full), Image.NEAREST)
    parse_full = Image.open(parse_p).resize((W_full, H_full), Image.NEAREST)

    pose_full = pose_p  # path for later

    # ─── 2) Build `opt` exactly as your segmentation/GMM/ALIAS were trained ────────
    seg_h, seg_w = 256, 192  # segmentation input resolution
    opt = SimpleNamespace(
        input_nc=13 + 8,  # 1 mask + 3 cloth + 13 parse + 3 pose + 1 noise = 21
        semantic_nc=13,
        load_height=seg_h,  # height for GMM grid generation & regression
        load_width=seg_w,  # width  "  "
        init_type="none",
        init_variance=0.02,
        grid_size=5,
        norm_G="spectralaliasinstance",
        ngf=64,
        num_upsampling_layers="most",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        TryOnModel(
            opt,
            os.path.join(args.checkpoints, "seg_final.pth"),
            os.path.join(args.checkpoints, "gmm_final.pth"),
            os.path.join(args.checkpoints, "alias_final.pth"),
        )
        .to(device)
        .eval()
    )

    # ─── 3) Resize down to seg/GMM resolution and tensorize ────────────────────────
    person_ds = person_full.resize((seg_w, seg_h), Image.BILINEAR)
    cloth_ds = cloth_full.resize((seg_w, seg_h), Image.BILINEAR)
    mask_ds = mask_full.resize((seg_w, seg_h), Image.NEAREST)
    parse_ds = parse_full.resize((seg_w, seg_h), Image.NEAREST)
    pose_ds = load_pose_render(pose_p, (seg_w, seg_h))

    # to tensors
    person_t = image_to_tensor(person_ds, True).to(device)  # [1,3,192,256]
    cloth_t = image_to_tensor(cloth_ds, True).to(device)  # [1,3,192,256]
    mask_t = image_to_tensor(mask_ds, True).to(device)  # [1,1,192,256]

    # one-hot parse: (13,H,W) → [1,13,H,W]
    parse_np = np.array(parse_ds, dtype=np.int64)
    print("unique labels in parse_np:", np.unique(parse_np))
    parse_oh = np.stack([(parse_np == c).astype(np.uint8) for c in range(13)], axis=0)
    # parse_t = torch.from_numpy(parse_oh).float().unsqueeze(0).to(device)
    parse_t = torch.from_numpy(parse_oh).float().unsqueeze(0)
    # map {0,1} → {–1,+1}
    parse_t = parse_t.mul(2.0).sub(1.0).to(device)

    # pose_t = pose_ds.to(device)  # [1,3,192,256]
    pose_t = load_pose_render(pose_p, (seg_w, seg_h)).to(device)
    pose_t = pose_t.mul(2.0).sub(1.0)

    # ─── 4) Inference ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model.infer(person_t, cloth_t, mask_t, parse_t, pose_t)
        # out is [1,3,192,256] in [-1,1], we map back to [0,1]
        out_img = (out.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        transforms.ToPILImage()(out_img).save(args.out)

    print(f"Saved try-on result to {args.out}")


if __name__ == "__main__":
    main()
