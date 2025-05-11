# #!/usr/bin/env python3
# """
# Human parsing utility using Self-Correction Human Parsing (SCHP).

# Provides:
#   - parse_human(img_array: np.ndarray, model_restore: str, dataset: str='lip') -> np.ndarray
#       Runs SCHP on a single image (as numpy array) and returns the parse map (H×W) as a numpy array.

#   - Batch CLI: python src/preprocessing/parse_human.py --input-dir INPUT --output-dir OUTPUT \
#         --model-restore PATH_TO_CHECKPOINT [--dataset lip|atr|pascal]
#     Wraps the SCHP simple_extractor.py script, which must be cloned at the project root under
#     'Self-Correction-Human-Parsing/'.
# """
# import os
# import argparse
# import tempfile
# import shutil
# import subprocess
# import uuid
# from PIL import Image
# import numpy as np


# def parse_human(
#     img_array: np.ndarray, model_restore: str, dataset: str = "lip"
# ) -> np.ndarray:
#     """
#     Parse a single image array using SCHP and return a 2D segmentation map.

#     Args:
#         img_array: H×W×3 uint8 numpy array (RGB).
#         model_restore: path to SCHP checkpoint (e.g. 'checkpoints/final.pth').
#         dataset: which SCHP label set ('lip', 'atr', or 'pascal').

#     Returns:
#         parse_map: H×W uint8 array where each pixel is the class index [0..N-1].
#     """
#     # Locate the simple_extractor script (assumes repo is at project_root/Self-Correction-Human-Parsing)
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
#     extractor = os.path.join(
#         project_root, "Self-Correction-Human-Parsing", "simple_extractor.py"
#     )
#     if not os.path.exists(extractor):
#         raise FileNotFoundError(f"Cannot find SCHP extractor script at {extractor}")

#     # Prepare temp dirs
#     tmp = tempfile.mkdtemp(prefix="schp_")
#     in_dir = os.path.join(tmp, "in")
#     out_dir = os.path.join(tmp, "out")
#     os.makedirs(in_dir, exist_ok=True)
#     os.makedirs(out_dir, exist_ok=True)

#     # Save input image
#     in_path = os.path.join(in_dir, f"{uuid.uuid4().hex}.jpg")
#     Image.fromarray(img_array).save(in_path)

#     # Invoke SCHP simple_extractor
#     cmd = [
#         "python",
#         extractor,
#         "--dataset",
#         dataset,
#         "--model-restore",
#         model_restore,
#         "--input-dir",
#         in_dir,
#         "--output-dir",
#         out_dir,
#     ]
#     subprocess.run(cmd, check=True)

#     # Read the output (assumes single file)
#     files = os.listdir(out_dir)
#     if not files:
#         shutil.rmtree(tmp)
#         raise RuntimeError(f"No output parsed files in {out_dir}")
#     out_file = files[0]
#     parse_img = Image.open(os.path.join(out_dir, out_file)).convert("L")
#     parse_map = np.array(parse_img, dtype=np.uint8)

#     # Cleanup
#     shutil.rmtree(tmp)
#     return parse_map


# def batch_parse(input_dir: str, output_dir: str, model_restore: str, dataset: str):
#     """
#     Batch parse all images in input_dir to output_dir using SCHP.
#     """
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
#     extractor = os.path.join(
#         project_root, "Self-Correction-Human-Parsing", "simple_extractor.py"
#     )

#     cmd = [
#         "python",
#         extractor,
#         "--dataset",
#         dataset,
#         "--model-restore",
#         model_restore,
#         "--input-dir",
#         input_dir,
#         "--output-dir",
#         output_dir,
#     ]
#     subprocess.run(cmd, check=True)
#     print(f"Batch parsing complete. Outputs in {output_dir}")


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Batch human parsing with SCHP simple_extractor"
#     )
#     parser.add_argument(
#         "--input-dir", "-i", required=True, help="Directory of input images"
#     )
#     parser.add_argument(
#         "--output-dir", "-o", required=True, help="Directory for parsed output maps"
#     )
#     parser.add_argument(
#         "--model-restore",
#         "-m",
#         required=True,
#         help="Path to SCHP checkpoint (final.pth)",
#     )
#     parser.add_argument(
#         "--dataset",
#         "-d",
#         choices=["lip", "atr", "pascal"],
#         default="lip",
#         help="Label set to use",
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     batch_parse(args.input_dir, args.output_dir, args.model_restore, args.dataset)


# if __name__ == "__main__":
#     main()


# src/preprocessing/parse_human.py
# src/preprocessing/parse_human.py

import torch, numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _model
    if _model is None:
        _model = deeplabv3_resnet101(pretrained=True).to(_device).eval()
    return _model


def parse_human(img_array: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Binary person mask via torchvision DeepLabV3 (COCO person = class 15).
    """
    model = _load_model()
    img = Image.fromarray(img_array)
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        out = model(inp)["out"][0]
        labels = out.argmax(0).cpu().numpy()
    return (labels == 15).astype(np.uint8)
