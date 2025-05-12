# #!/usr/bin/env python3
# """
# End-to-end inference script: preprocess a single person+cloth pair, run the try-on model via the copied test.py,
# and output the final composite image.

# Usage:
#   python run_inference.py \
#     --person PERSON.jpg \
#     --cloth CLOTH.jpg \
#     --checkpoint-dir ./checkpoints \
#     --out OUTPUT.png
# """
# import os
# import sys
# import argparse
# import tempfile
# import shutil
# import subprocess

# # Allow imports from src/
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(SCRIPT_DIR, "src")
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)

# from preprocessing.runner import run_all


# def main():
#     parser = argparse.ArgumentParser(
#         description="Full pipeline: preprocess → inference → save output"
#     )
#     parser.add_argument("--person", "-p", required=True, help="Path to person image")
#     parser.add_argument("--cloth", "-c", required=True, help="Path to cloth image")
#     parser.add_argument(
#         "--checkpoint-dir",
#         "-k",
#         default="./checkpoints",
#         help="Directory containing all model checkpoints",
#     )
#     parser.add_argument("--out", "-o", required=True, help="Path for final output PNG")
#     # Preprocessing args
#     parser.add_argument("--width", type=int, default=768, help="Resize width")
#     parser.add_argument("--height", type=int, default=1024, help="Resize height")
#     parser.add_argument(
#         "--resample",
#         choices=["nearest", "bilinear", "bicubic", "lanczos"],
#         default="bilinear",
#         help="Resize filter",
#     )
#     parser.add_argument(
#         "--mask-ckpt",
#         default="./checkpoints/cloth_segm_u2net_latest.pth",
#         help="U²-Net mask checkpoint",
#     )
#     parser.add_argument(
#         "--parse-ckpt",
#         default="./checkpoints/final.pth",
#         help="SCHP parsing checkpoint",
#     )
#     parser.add_argument(
#         "--parse-dataset",
#         choices=["lip", "atr", "pascal"],
#         default="lip",
#         help="SCHP label set to use",
#     )
#     parser.add_argument(
#         "--pose-backend",
#         choices=["openpose", "mediapipe"],
#         default="mediapipe",
#         help="Pose estimation backend",
#     )
#     parser.add_argument(
#         "--openpose-bin",
#         default=None,
#         help="Path to OpenPose binary (if using openpose)",
#     )
#     args = parser.parse_args()

#     # Create a temporary workspace
#     tmp = tempfile.mkdtemp(prefix="vton_")
#     inputs_root = os.path.join(tmp, "inputs")
#     test_root = os.path.join(inputs_root, "test")
#     os.makedirs(test_root, exist_ok=True)

#     # Step 1: Preprocessing
#     run_all(
#         person_path=args.person,
#         cloth_path=args.cloth,
#         out_dir=test_root,
#         width=args.width,
#         height=args.height,
#         resample=args.resample,
#         mask_ckpt=args.mask_ckpt,
#         parse_ckpt=args.parse_ckpt,
#         parse_dataset=args.parse_dataset,
#         pose_backend=args.pose_backend,
#         openpose_bin=args.openpose_bin,
#     )

#     # Step 2: create test_pairs file for test.py
#     pairs_file = os.path.join(inputs_root, "test_pairs.txt")
#     with open(pairs_file, "w") as f:
#         f.write(f"{os.path.basename(args.person)} {os.path.basename(args.cloth)}")

#     # Step 3: run the copied test.py in this repo
#     test_py = os.path.join(SCRIPT_DIR, "test.py")
#     if not os.path.exists(test_py):
#         raise FileNotFoundError(f"test.py not found in {SCRIPT_DIR}")
#     cmd = [
#         sys.executable,
#         test_py,
#         "--name",
#         "output",
#         "--dataset_dir",
#         inputs_root,
#         "--checkpoint_dir",
#         args.checkpoint_dir,
#         "--save_dir",
#         tmp,
#     ]
#     print("Running inference:", " ".join(cmd))
#     subprocess.run(cmd, check=True)

#     # Step 4: gather the output image
#     out_folder = os.path.join(tmp, "output")
#     if not os.path.isdir(out_folder):
#         raise RuntimeError(f"Expected output folder not found: {out_folder}")
#     imgs = sorted(os.listdir(out_folder))
#     if not imgs:
#         raise RuntimeError(f"No images in output folder {out_folder}")
#     src_img = os.path.join(out_folder, imgs[0])
#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     shutil.copy(src_img, args.out)
#     print(f"Saved final try-on image to {args.out}")

#     # Cleanup temporary files
#     shutil.rmtree(tmp)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
End-to-end inference script: preprocess a single person+cloth pair, run the try-on model via the copied test.py,
and output the final composite image.

Usage:
  python run_inference.py \
    --person PERSON.jpg \
    --cloth CLOTH.jpg \
    --checkpoint-dir ./checkpoints \
    --out OUTPUT.png
"""
import os
import sys
import argparse
import tempfile
import shutil
import subprocess

# Allow imports from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.preprocessing.runner import run_all


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: preprocess → inference → save output"
    )
    parser.add_argument("--person", "-p", required=True, help="Path to person image")
    parser.add_argument("--cloth", "-c", required=True, help="Path to cloth image")
    parser.add_argument(
        "--checkpoint-dir",
        "-k",
        default="./checkpoints",
        help="Directory containing all model checkpoints",
    )
    parser.add_argument("--out", "-o", required=True, help="Path for final output PNG")
    # Preprocessing args
    parser.add_argument("--width", type=int, default=768, help="Resize width")
    parser.add_argument("--height", type=int, default=1024, help="Resize height")
    parser.add_argument(
        "--resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bilinear",
        help="Resize filter",
    )
    parser.add_argument(
        "--mask-ckpt",
        default="./checkpoints/cloth_segm_u2net_latest.pth",
        help="U²-Net mask checkpoint",
    )
    parser.add_argument(
        "--parse-ckpt",
        default="./checkpoints/final.pth",
        help="SCHP parsing checkpoint",
    )
    parser.add_argument(
        "--parse-dataset",
        choices=["lip", "atr", "pascal"],
        default="lip",
        help="SCHP label set to use",
    )
    parser.add_argument(
        "--pose-backend",
        choices=["openpose", "mediapipe"],
        default="mediapipe",
        help="Pose estimation backend",
    )
    parser.add_argument(
        "--openpose-bin",
        default=None,
        help="Path to OpenPose binary (if using openpose)",
    )
    args = parser.parse_args()

    # Create a temporary workspace
    tmp = tempfile.mkdtemp(prefix="vton_")
    inputs_root = os.path.join(tmp, "inputs")
    test_root = os.path.join(inputs_root, "test")
    os.makedirs(test_root, exist_ok=True)

    # Step 1: Preprocessing
    run_all(
        person_path=args.person,
        cloth_path=args.cloth,
        out_dir=test_root,
        width=args.width,
        height=args.height,
        resample=args.resample,
        mask_ckpt=args.mask_ckpt,
        parse_ckpt=args.parse_ckpt,
        parse_dataset=args.parse_dataset,
        pose_backend=args.pose_backend,
        openpose_bin=args.openpose_bin,
    )

    # Step 2: create test_pairs file for test.py
    pairs_file = os.path.join(inputs_root, "test_pairs.txt")
    with open(pairs_file, "w") as f:
        f.write(f"{os.path.basename(args.person)} {os.path.basename(args.cloth)}")

    # Step 3: run the copied test.py in this repo
    # test_py = os.path.join(SCRIPT_DIR, "test.py")
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    test_py = os.path.join(PROJECT_ROOT, "test.py")

    if not os.path.exists(test_py):
        raise FileNotFoundError(f"test.py not found in {SCRIPT_DIR}")
    cmd = [
        sys.executable,
        test_py,
        "--name",
        "output",
        "--dataset_dir",
        inputs_root,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--save_dir",
        tmp,
    ]
    print("Running inference:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Step 4: gather the output image
    out_folder = os.path.join(tmp, "output")
    if not os.path.isdir(out_folder):
        raise RuntimeError(f"Expected output folder not found: {out_folder}")
    imgs = sorted(os.listdir(out_folder))
    if not imgs:
        raise RuntimeError(f"No images in output folder {out_folder}")
    src_img = os.path.join(out_folder, imgs[0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy(src_img, args.out)
    print(f"Saved final try-on image to {args.out}")

    # Cleanup temporary files
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
