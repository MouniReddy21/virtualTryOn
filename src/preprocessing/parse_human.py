#!/usr/bin/env python3
"""
parse_human.py: unified human parsing interface supporting DeepLabV3 or SCHP (CIHP/ATR/Pascal).
"""
import os
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Globals for caching
_deeplab_model = None
_schp_checkpoint = None

# Base directory of SCHP repo (adjust as needed)
SCHP_ROOT = os.getenv("SCHP_ROOT", "./Self-Correction-Human-Parsing")


# DeepLabV3 loader
def _load_deeplab(device):
    global _deeplab_model
    if _deeplab_model is None:
        model = deeplabv3_resnet101(pretrained=True)
        model = model.to(device).eval()
        _deeplab_model = model
    return _deeplab_model


# Parse human using DeepLabV3 (binary mask)
def _parse_deeplab(img_array):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_deeplab(device)
    img = Image.fromarray(img_array)
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)["out"][0]
        labels = out.argmax(0).cpu().numpy()
    # binary: person class == 15
    return (labels == 15).astype(np.uint8)


# Parse human using SCHP via simple_extractor.py
# dataset should be one of: lip, atr, cihp, pascal
# schp_checkpoint: path to SCHP .pth
# caches outputs in memory via subprocess invocation per image
_schp_tmpdir = None


def _parse_schp(img_path, dataset, schp_checkpoint):
    global _schp_tmpdir
    if _schp_tmpdir is None:
        _schp_tmpdir = tempfile.mkdtemp(prefix="schp_parse_")
    # call simple_extractor on single image
    cmd = [
        "python3",
        os.path.join(SCHP_ROOT, "simple_extractor.py"),
        "--dataset",
        dataset,
        "--model-restore",
        schp_checkpoint,
        "--input-dir",
        os.path.dirname(img_path),
        "--output-dir",
        _schp_tmpdir,
    ]
    subprocess.run(cmd, check=True)
    # read corresponding PNG
    base = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    out_png = os.path.join(_schp_tmpdir, base)
    mask = Image.open(out_png)
    return np.array(mask, dtype=np.uint8)


# Public API
def parse_human(
    img_array: np.ndarray, parse_ckpt: str, parse_dataset: str
) -> np.ndarray:
    """
    img_array: HxWx3 uint8 array
    parse_ckpt: path to checkpoint (.pth)
    parse_dataset: one of ['lip','atr','cihp','pascal','deeplab']

    returns: HxW array of labels (uint8)
      - for 'deeplab': binary mask (0/1)
      - for others: multi-class mask with values 0..C-1
    """
    # Save img to temporary file
    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(img_array).save(tmp_img.name)
    if parse_dataset.lower() in ["deeplab"]:
        lbl = _parse_deeplab(img_array)
    else:
        # SCHP backend
        lbl = _parse_schp(tmp_img.name, parse_dataset, parse_ckpt)
    # cleanup temp image
    os.unlink(tmp_img.name)
    return lbl
