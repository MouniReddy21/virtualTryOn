# #!/usr/bin/env python3
# """
# runner.py: orchestrates the full preprocessing pipeline for a cloth-person pair.

# Steps:
#  1. Copy & resize cloth & person images
#  2. Generate cloth mask via U²-Net
#  3. Remove background from person via rembg
#  4. Parse human segmentation via DeepLabV3-based parse_human
#  5. Estimate pose via MediaPipe or OpenPose

# Outputs all artifacts under a single --out-dir, preserving the structure:
#     out-dir/
#       ├── cloth/
#       ├── image/
#       ├── cloth-mask/
#       ├── image-parse/
#       ├── openpose-img/
#       └── openpose-json/
# """
# import os
# import argparse
# import shutil
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms

# # Relative imports from preprocessing modules
# from .resize import resize_image_file
# from .cloth_mask import load_model as load_mask_model, generate_mask
# from .remove_bg import remove_background
# from .parse_human import parse_human
# from .detect_pose import run_openpose, run_mediapipe


# def get_resample_filter(name: str):
#     return {
#         "nearest": Image.NEAREST,
#         "bilinear": Image.BILINEAR,
#         "bicubic": Image.BICUBIC,
#         "lanczos": Image.LANCZOS,
#     }[name]


# def run_all(
#     person_path: str,
#     cloth_path: str,
#     out_dir: str,
#     width: int,
#     height: int,
#     resample: str,
#     mask_ckpt: str,
#     parse_ckpt: str,
#     parse_dataset: str,
#     pose_backend: str,
#     openpose_bin: str | None,
# ) -> None:
#     # 1. Prepare output subdirectories
#     cloth_dir = os.path.join(out_dir, "cloth")
#     image_dir = os.path.join(out_dir, "image")
#     cloth_mask_dir = os.path.join(out_dir, "cloth-mask")
#     image_parse_dir = os.path.join(out_dir, "image-parse")
#     openpose_img_dir = os.path.join(out_dir, "openpose-img")
#     openpose_json_dir = os.path.join(out_dir, "openpose-json")

#     for d in [
#         cloth_dir,
#         image_dir,
#         cloth_mask_dir,
#         image_parse_dir,
#         openpose_img_dir,
#         openpose_json_dir,
#     ]:
#         os.makedirs(d, exist_ok=True)

#     # Copy originals
#     cloth_fname = os.path.basename(cloth_path)
#     image_fname = os.path.basename(person_path)
#     shutil.copy2(cloth_path, os.path.join(cloth_dir, cloth_fname))
#     shutil.copy2(person_path, os.path.join(image_dir, image_fname))

#     # 2. Resize cloth & person
#     resample_filter = get_resample_filter(resample)
#     resize_image_file(
#         os.path.join(cloth_dir, cloth_fname),
#         os.path.join(cloth_dir, cloth_fname),
#         (width, height),
#         resample_filter,
#     )
#     resize_image_file(
#         os.path.join(image_dir, image_fname),
#         os.path.join(image_dir, image_fname),
#         (width, height),
#         resample_filter,
#     )

#     # 3. Generate cloth mask
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
#     mask_net = load_mask_model(mask_ckpt, device)
#     mask_transform = transforms.Compose(
#         [
#             transforms.Resize((height, width)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5] * 3, [0.5] * 3),
#         ]
#     )
#     mask = generate_mask(
#         mask_net, os.path.join(cloth_dir, cloth_fname), mask_transform, device
#     )
#     mask.save(os.path.join(cloth_mask_dir, os.path.splitext(cloth_fname)[0] + ".png"))
#     print("Cloth mask saved.")

#     # 4. Remove background from person
#     print("Removing background from person image...")
#     rgba = remove_background(os.path.join(image_dir, image_fname))
#     # Convert RGBA to RGB before saving as JPEG
#     img_rgba = Image.fromarray(rgba)
#     img_rgb = img_rgba.convert("RGB")
#     img_rgb.save(os.path.join(image_dir, image_fname))

#     # 5. Human parsing
#     print("Parsing human segmentation...")
#     img_arr = np.array(img_rgb, dtype=np.uint8)
#     parse_map = parse_human(img_arr, parse_ckpt, parse_dataset)
#     Image.fromarray((parse_map * 255).astype("uint8")).save(
#         os.path.join(image_parse_dir, os.path.splitext(image_fname)[0] + ".png")
#     )

#     # 6. Pose estimation
#     print(f"Estimating pose via {pose_backend}...")
#     if pose_backend == "openpose":
#         if not openpose_bin:
#             raise ValueError("`--openpose-bin` is required for the openpose backend")
#         run_openpose(image_dir, openpose_json_dir, openpose_img_dir, openpose_bin)
#     else:
#         run_mediapipe(image_dir, openpose_json_dir, openpose_img_dir)

#     print(f"Preprocessing complete. All artifacts in {out_dir}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Full preprocessing runner for virtual try-on"
#     )
#     parser.add_argument("--person", "-p", required=True, help="Path to person image")
#     parser.add_argument("--cloth", "-c", required=True, help="Path to cloth image")
#     parser.add_argument(
#         "--out-dir", "-o", required=True, help="Directory to emit outputs"
#     )
#     parser.add_argument("--width", type=int, default=768, help="Resize width")
#     parser.add_argument("--height", type=int, default=1024, help="Resize height")
#     parser.add_argument(
#         "--resample",
#         choices=["nearest", "bilinear", "bicubic", "lanczos"],
#         default="bilinear",
#     )
#     parser.add_argument(
#         "--mask-ckpt", default="checkpoints/cloth_segm_u2net_latest.pth"
#     )
#     parser.add_argument("--parse-ckpt", default="checkpoints/final.pth")
#     parser.add_argument(
#         "--parse-dataset", choices=["lip", "atr", "pascal"], default="lip"
#     )
#     parser.add_argument(
#         "--pose-backend", choices=["openpose", "mediapipe"], default="mediapipe"
#     )
#     parser.add_argument(
#         "--openpose-bin", help="Path to OpenPose binary (if using openpose)"
#     )
#     args = parser.parse_args()
#     run_all(
#         args.person,
#         args.cloth,
#         args.out_dir,
#         args.width,
#         args.height,
#         args.resample,
#         args.mask_ckpt,
#         args.parse_ckpt,
#         args.parse_dataset,
#         args.pose_backend,
#         args.openpose_bin,
#     )


# if __name__ == "__main__":
#     main()
# ------------------------------------------------------------------------------


# #!/usr/bin/env python3
# """
# runner.py: orchestrates the full preprocessing pipeline for a cloth-person pair.

# Steps:
#  1. Copy & resize cloth & person images
#  2. Generate cloth mask via U²-Net
#  3. Remove background from person via rembg
#  4. Parse human segmentation via DeepLabV3-based parse_human
#  5. Estimate pose via MediaPipe or OpenPose

# Outputs all artifacts under a single --out-dir, preserving the structure:
#     out-dir/
#       ├── cloth/
#       ├── image/
#       ├── cloth-mask/
#       ├── image-parse/
#       ├── openpose-img/
#       └── openpose-json/
# """
# import os
# import argparse
# import shutil
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms

# # Relative imports from preprocessing modules
# from .resize import resize_image_file
# from .cloth_mask import load_model as load_mask_model, generate_mask
# from .remove_bg import remove_background
# from .parse_human import parse_human
# from .detect_pose import run_openpose, run_mediapipe

# import subprocess, shutil, os


# # SCHP_ROOT = os.getenv("SCHP_ROOT", os.path.abspath(os.path.join(os.getcwd(), "Self-Correction-Human-Parsing")))


# def get_resample_filter(name: str):
#     return {
#         "nearest": Image.NEAREST,
#         "bilinear": Image.BILINEAR,
#         "bicubic": Image.BICUBIC,
#         "lanczos": Image.LANCZOS,
#     }[name]


# def run_all(
#     person_path: str,
#     cloth_path: str,
#     out_dir: str,
#     width: int,
#     height: int,
#     resample: str,
#     mask_ckpt: str,
#     parse_ckpt: str,
#     parse_dataset: str,
#     pose_backend: str,
#     openpose_bin: str | None,
# ) -> None:
#     # 1. Prepare output subdirectories
#     cloth_dir = os.path.join(out_dir, "cloth")
#     image_dir = os.path.join(out_dir, "image")
#     cloth_mask_dir = os.path.join(out_dir, "cloth-mask")
#     image_parse_dir = os.path.join(out_dir, "image-parse")
#     openpose_img_dir = os.path.join(out_dir, "openpose-img")
#     openpose_json_dir = os.path.join(out_dir, "openpose-json")

#     for d in [
#         cloth_dir,
#         image_dir,
#         cloth_mask_dir,
#         image_parse_dir,
#         openpose_img_dir,
#         openpose_json_dir,
#     ]:
#         os.makedirs(d, exist_ok=True)

#     # Copy originals
#     cloth_fname = os.path.basename(cloth_path)
#     image_fname = os.path.basename(person_path)
#     shutil.copy2(cloth_path, os.path.join(cloth_dir, cloth_fname))
#     shutil.copy2(person_path, os.path.join(image_dir, image_fname))

#     # 2. Resize cloth & person
#     resample_filter = get_resample_filter(resample)
#     resize_image_file(
#         os.path.join(cloth_dir, cloth_fname),
#         os.path.join(cloth_dir, cloth_fname),
#         (width, height),
#         resample_filter,
#     )
#     resize_image_file(
#         os.path.join(image_dir, image_fname),
#         os.path.join(image_dir, image_fname),
#         (width, height),
#         resample_filter,
#     )

#     # 3. Generate cloth mask
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
#     mask_net = load_mask_model(mask_ckpt, device)
#     mask_transform = transforms.Compose(
#         [
#             transforms.Resize((height, width)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5] * 3, [0.5] * 3),
#         ]
#     )
#     mask = generate_mask(
#         mask_net, os.path.join(cloth_dir, cloth_fname), mask_transform, device
#     )
#     mask.save(os.path.join(cloth_mask_dir, os.path.splitext(cloth_fname)[0] + ".png"))
#     print("Cloth mask saved.")

#     # 4. Remove background from person
#     print("Removing background from person image...")
#     # rgba = remove_background(os.path.join(image_dir, image_fname)) # temporary comment, for gpu enable
#     print("→ background removal done")
#     # Convert RGBA to RGB before saving as JPEG
#     # img_rgba = Image.fromarray(rgba) # temporary comment for gpu enable
#     # img_rgb = img_rgba.convert("RGB") # temporary comment for gpu enable
#     img_rgb = Image.open(os.path.join(image_dir, image_fname)).convert(
#         "RGB"
#     )  # for gpu remove

#     img_rgb.save(os.path.join(image_dir, image_fname))

#     # 5. Human parsing
#     print("Parsing human segmentation...")
#     if parse_dataset in ("lip", "atr", "cihp", "pascal"):
#         # 5a) Run SCHP once for the whole image folder:
#         # schp_out = os.path.join(out_dir, "schp-parse")
#         # os.makedirs(schp_out, exist_ok=True)
#         # cmd = [
#         #     "python3",
#         #     os.path.join(SCHP_ROOT, "simple_extractor.py"),
#         #     "--dataset",
#         #     parse_dataset,
#         #     "--model-restore",
#         #     parse_ckpt,
#         #     "--input-dir",
#         #     image_dir,
#         #     "--output-dir",
#         #     schp_out,
#         # ]
#         # subprocess.run(cmd, check=True)
#         # # 5b) Copy SCHP outputs into image-parse/
#         # for fname in os.listdir(schp_out):
#         #     shutil.copy2(
#         #         os.path.join(schp_out, fname), os.path.join(image_parse_dir, fname)
#         #     )
#         for fname in os.listdir(image_dir):
#         basename = os.path.splitext(fname)[0] + ".png"
#         src = os.path.join(image_dir.replace("image","image-parse"), basename)
#         dst = os.path.join(image_parse_dir, basename)
#         shutil.copy2(src, dst)
#     else:
#         # DeepLabV3 or binary fallback
#         img_arr = np.array(img_rgb, dtype=np.uint8)
#         parse_map = parse_human(img_arr, parse_ckpt, parse_dataset)
#     Image.fromarray((parse_map * 255).astype("uint8")).save(
#         os.path.join(image_parse_dir, os.path.splitext(image_fname)[0] + ".png")
#     )

#     # 6. Pose estimation
#     print(f"Estimating pose via {pose_backend}...")
#     if pose_backend == "openpose":
#         if not openpose_bin:
#             raise ValueError("`--openpose-bin` is required for the openpose backend")
#         run_openpose(image_dir, openpose_json_dir, openpose_img_dir, openpose_bin)
#     else:
#         run_mediapipe(image_dir, openpose_json_dir, openpose_img_dir)

#     print(f"Preprocessing complete. All artifacts in {out_dir}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Full preprocessing runner for virtual try-on"
#     )
#     parser.add_argument("--person", "-p", required=True, help="Path to person image")
#     parser.add_argument("--cloth", "-c", required=True, help="Path to cloth image")
#     parser.add_argument(
#         "--out-dir", "-o", required=True, help="Directory to emit outputs"
#     )
#     parser.add_argument("--width", type=int, default=768, help="Resize width")
#     parser.add_argument("--height", type=int, default=1024, help="Resize height")
#     parser.add_argument(
#         "--resample",
#         choices=["nearest", "bilinear", "bicubic", "lanczos"],
#         default="bilinear",
#     )
#     parser.add_argument(
#         "--mask-ckpt", default="checkpoints/cloth_segm_u2net_latest.pth"
#     )
#     parser.add_argument("--parse-ckpt", default="checkpoints/final.pth")
#     parser.add_argument(
#         "--parse-dataset", choices=["lip", "atr", "pascal"], default="lip"
#     )
#     parser.add_argument(
#         "--pose-backend", choices=["openpose", "mediapipe"], default="mediapipe"
#     )
#     parser.add_argument(
#         "--openpose-bin", help="Path to OpenPose binary (if using openpose)"
#     )
#     args = parser.parse_args()
#     run_all(
#         args.person,
#         args.cloth,
#         args.out_dir,
#         args.width,
#         args.height,
#         args.resample,
#         args.mask_ckpt,
#         args.parse_ckpt,
#         args.parse_dataset,
#         args.pose_backend,
#         args.openpose_bin,
#     )


# if __name__ == "__main__":
#     main()


# # -----------------------
# #!/usr/bin/env python3
# """
# runner.py: orchestrates the full preprocessing pipeline for a cloth-person pair,
# with optional SCHP precomputation and reuse of masks.
# """
# import os
# import argparse
# import shutil
# import subprocess
# import tempfile
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms

# # Relative imports from preprocessing modules
# from .resize import resize_image_file
# from .cloth_mask import load_model as load_mask_model, generate_mask
# from .remove_bg import remove_background
# from .parse_human import parse_human
# from .detect_pose import run_openpose, run_mediapipe


# # Default SCHP repo location (override via env or CLI)
# def default_schp_root():
#     return os.getenv(
#         "SCHP_ROOT",
#         os.path.abspath(os.path.join(os.getcwd(), "Self-Correction-Human-Parsing")),
#     )


# def get_resample_filter(name: str):
#     return {
#         "nearest": Image.NEAREST,
#         "bilinear": Image.BILINEAR,
#         "bicubic": Image.BICUBIC,
#         "lanczos": Image.LANCZOS,
#     }[name]


# def run_all(
#     person_path: str,
#     cloth_path: str,
#     out_dir: str,
#     width: int,
#     height: int,
#     resample: str,
#     mask_ckpt: str,
#     parse_ckpt: str,
#     parse_dataset: str,
#     pose_backend: str,
#     openpose_bin: str | None,
#     schp_root: str,
#     # precomputed_masks: str | None,
#     precomputed_cloth_masks: str | None,
#     # precomputed_schp_masks: str | None,
#     precomputed_parse_masks: str | None,
#     precomputed_pose_img: str | None,
#     precomputed_pose_json: str | None,
# ) -> None:
#     # 1. Prepare output subdirectories
#     cloth_dir = os.path.join(out_dir, "cloth")
#     image_dir = os.path.join(out_dir, "image")
#     cloth_mask_dir = os.path.join(out_dir, "cloth-mask")
#     image_parse_dir = os.path.join(out_dir, "image-parse")
#     openpose_img_dir = os.path.join(out_dir, "openpose-img")
#     openpose_json_dir = os.path.join(out_dir, "openpose-json")

#     for d in [
#         cloth_dir,
#         image_dir,
#         cloth_mask_dir,
#         image_parse_dir,
#         openpose_img_dir,
#         openpose_json_dir,
#     ]:
#         os.makedirs(d, exist_ok=True)

#     # Copy originals
#     cloth_fname = os.path.basename(cloth_path)
#     image_fname = os.path.basename(person_path)
#     shutil.copy2(cloth_path, os.path.join(cloth_dir, cloth_fname))
#     shutil.copy2(person_path, os.path.join(image_dir, image_fname))

#     # 2. Resize cloth & person
#     resample_filter = get_resample_filter(resample)
#     resize_image_file(
#         os.path.join(cloth_dir, cloth_fname),
#         os.path.join(cloth_dir, cloth_fname),
#         (width, height),
#         resample_filter,
#     )
#     resize_image_file(
#         os.path.join(image_dir, image_fname),
#         os.path.join(image_dir, image_fname),
#         (width, height),
#         resample_filter,
#     )

#     # # 3. Generate cloth mask
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
#     # mask_net = load_mask_model(mask_ckpt, device)
#     # mask_transform = transforms.Compose(
#     #     [
#     #         transforms.Resize((height, width)),
#     #         transforms.ToTensor(),
#     #         transforms.Normalize([0.5] * 3, [0.5] * 3),
#     #     ]
#     # )
#     # mask = generate_mask(
#     #     mask_net, os.path.join(cloth_dir, cloth_fname), mask_transform, device
#     # )
#     # mask.save(os.path.join(cloth_mask_dir, os.path.splitext(cloth_fname)[0] + ".png"))
#     # print("Cloth mask saved.")

#     # 3. Generate (or copy) cloth mask
#     # 3. Generate or copy cloth mask

#     stem = os.path.splitext(cloth_fname)[0]
#     dst = os.path.join(cloth_mask_dir, stem + ".png")

#     if precomputed_cloth_masks:
#         src = os.path.join(precomputed_cloth_masks, stem + ".jpg")
#         if not os.path.exists(src):
#             raise FileNotFoundError(f"No mask at {src}")
#         shutil.copy2(src, dst)
#         print(f"Copied precomputed cloth mask for {stem}")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
#         mask_net = load_mask_model(mask_ckpt, device)
#         mask_transform = transforms.Compose(
#             [
#                 transforms.Resize((height, width)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5] * 3, [0.5] * 3),
#             ]
#         )
#         mask = generate_mask(
#             mask_net, os.path.join(cloth_dir, cloth_fname), mask_transform, device
#         )
#         mask.save(dst)
#         print(f"Generated cloth mask for {stem}")

#     # 4. Remove background from person
#     print("Removing background from person image...")
#     # For GPU-enabled, uncomment remove_background
#     # rgba = remove_background(os.path.join(image_dir, image_fname))
#     print("→ background removal done")
#     img_rgb = Image.open(os.path.join(image_dir, image_fname)).convert("RGB")
#     img_rgb.save(os.path.join(image_dir, image_fname))

#     # 5. Human parsing
#     print("Parsing human segmentation...")
#     stem = os.path.splitext(image_fname)[0]
#     out_mask = os.path.join(image_parse_dir, stem + ".png")

#     if parse_dataset in ("lip", "atr", "cihp", "pascal"):
#         # 5a) Use precomputed if available
#         if precomputed_masks:
#             src = os.path.join(precomputed_masks, stem + ".jpg")
#             if os.path.exists(src):
#                 shutil.copy2(src, out_mask)
#                 print(f"Copied precomputed mask for {stem}")
#             else:
#                 raise FileNotFoundError(f"Mask not found in precomputed folder: {src}")
#         else:
#             # 5b) Run SCHP extractor once for this folder
#             schp_out = os.path.join(out_dir, "schp-parse")
#             os.makedirs(schp_out, exist_ok=True)
#             cmd = [
#                 "python3",
#                 os.path.join(schp_root, "simple_extractor.py"),
#                 "--dataset",
#                 parse_dataset,
#                 "--model-restore",
#                 parse_ckpt,
#                 "--input-dir",
#                 image_dir,
#                 "--output-dir",
#                 schp_out,
#             ]
#             subprocess.run(cmd, check=True)
#             # copy single
#             tmp = os.path.join(schp_out, stem + ".png")
#             shutil.copy2(tmp, out_mask)
#     else:
#         # DeepLabV3 fallback
#         arr = np.array(img_rgb, dtype=np.uint8)
#         parse_map = parse_human(arr, parse_ckpt, parse_dataset)
#         # Image.fromarray((parse_map * 255).astype("uint8")).save(out_mask)
#         Image.fromarray(parse_map.astype("uint8")).save(out_mask)

#     # 6. Pose estimation
#     print(f"Estimating pose via {pose_backend}...")
#     if precomputed_pose_img and precomputed_pose_json:
#         # copy *.png → openpose_img_dir
#         for f in sorted(os.listdir(precomputed_pose_img)):
#             shutil.copy2(
#                 os.path.join(precomputed_pose_img, f),
#                 os.path.join(openpose_img_dir, f),
#             )
#         # copy *.json → openpose_json_dir
#         for f in sorted(os.listdir(precomputed_pose_json)):
#             shutil.copy2(
#                 os.path.join(precomputed_pose_json, f),
#                 os.path.join(openpose_json_dir, f),
#             )
#     else:
#         if pose_backend == "openpose":
#             if not openpose_bin:
#                 raise ValueError("--openpose-bin is required for openpose backend")
#             run_openpose(image_dir, openpose_json_dir, openpose_img_dir, openpose_bin)
#         else:
#             run_mediapipe(image_dir, openpose_json_dir, openpose_img_dir)

#     print(f"Preprocessing complete. All artifacts in {out_dir}")


# if __name__ == "__main__":
#     p = argparse.ArgumentParser(
#         description="Full preprocessing runner for virtual try-on"
#     )
#     p.add_argument(
#         "--precomputed-cloth-mask-dir",
#         help="Directory of precomputed U²-Net cloth masks (.png)",
#     )
#     p.add_argument("--person", "-p", required=True)
#     p.add_argument("--cloth", "-c", required=True)
#     p.add_argument("--out-dir", "-o", required=True)
#     p.add_argument("--width", type=int, default=768)
#     p.add_argument("--height", type=int, default=1024)
#     p.add_argument(
#         "--resample",
#         choices=["nearest", "bilinear", "bicubic", "lanczos"],
#         default="bilinear",
#     )
#     p.add_argument("--mask-ckpt", default="checkpoints/cloth_segm_u2net_latest.pth")
#     p.add_argument("--parse-ckpt", default="checkpoints/final.pth")
#     p.add_argument(
#         "--parse-dataset",
#         choices=["lip", "atr", "cihp", "pascal", "deeplab"],
#         default="lip",
#     )
#     p.add_argument(
#         "--pose-backend", choices=["openpose", "mediapipe"], default="mediapipe"
#     )
#     p.add_argument("--openpose-bin", help="Path to OpenPose binary")
#     p.add_argument("--schp-root", default=default_schp_root(), help="Path to SCHP repo")
#     p.add_argument(
#         "--precomputed-masks",
#         "--precomputed-parse-masks",
#         dest="precomputed_masks",
#         help="Directory of precomputed SCHP masks to reuse",
#     )
#     p.add_argument(
#         "--precomputed-pose-img",
#         help="Path to a folder of precomputed pose‐overlay PNGs",
#     )
#     p.add_argument(
#         "--precomputed-pose-json", help="Path to a folder of precomputed OpenPose JSONs"
#     )

#     args = p.parse_args()

#     run_all(
#         args.person,
#         args.cloth,
#         args.out_dir,
#         args.width,
#         args.height,
#         args.resample,
#         args.mask_ckpt,
#         args.parse_ckpt,
#         args.parse_dataset,
#         args.pose_backend,
#         args.openpose_bin,
#         args.schp_root,
#         args.precomputed_masks,
#         args.precomputed_cloth_mask_dir,
#         args.precomputed_masks,
#         args.precomputed_pose_img,
#         args.precomputed_pose_json,
#     )


# # I’ve refactored runner.py so that:

# # You can supply --precomputed-masks (e.g. your Drive folder). If provided, it will copy existing SCHP masks directly.

# # Otherwise, for SCHP-backed parsing (lip/atr/cihp/pascal), it runs simple_extractor.py once over the image/ folder and then copies just the current image’s mask.

# # DeepLabV3 remains as the fallback for --parse-dataset=deeplab.

# # Added CLI flags --schp-root and --precomputed-masks for easy configuration.


#!/usr/bin/env python3
"""
runner.py: orchestrates the full preprocessing pipeline for a cloth-person pair,
with optional SCHP precomputation and reuse of masks.
"""
import os
import argparse
import shutil
import subprocess
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


def default_schp_root():
    """Default SCHP repo location (override via env or CLI)."""
    return os.getenv(
        "SCHP_ROOT",
        os.path.abspath(os.path.join(os.getcwd(), "Self-Correction-Human-Parsing")),
    )


def get_resample_filter(name: str):
    return {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
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
    openpose_bin: str | None,
    schp_root: str,
    precomputed_cloth_masks: str | None,
    precomputed_parse_masks: str | None,
    precomputed_pose_img: str | None,
    precomputed_pose_json: str | None,
) -> None:
    # 1. Prepare output subdirectories
    cloth_dir = os.path.join(out_dir, "cloth")
    image_dir = os.path.join(out_dir, "image")
    cloth_mask_dir = os.path.join(out_dir, "cloth-mask")
    image_parse_dir = os.path.join(out_dir, "image-parse")
    openpose_img_dir = os.path.join(out_dir, "openpose-img")
    openpose_json_dir = os.path.join(out_dir, "openpose-json")
    for d in [
        cloth_dir,
        image_dir,
        cloth_mask_dir,
        image_parse_dir,
        openpose_img_dir,
        openpose_json_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # Copy originals
    cloth_fname = os.path.basename(cloth_path)
    image_fname = os.path.basename(person_path)
    shutil.copy2(cloth_path, os.path.join(cloth_dir, cloth_fname))
    shutil.copy2(person_path, os.path.join(image_dir, image_fname))

    # 2. Resize cloth & person
    rf = get_resample_filter(resample)
    resize_image_file(
        os.path.join(cloth_dir, cloth_fname),
        os.path.join(cloth_dir, cloth_fname),
        (width, height),
        rf,
    )
    resize_image_file(
        os.path.join(image_dir, image_fname),
        os.path.join(image_dir, image_fname),
        (width, height),
        rf,
    )

    # 3. Cloth mask (generate or copy)
    stem = os.path.splitext(cloth_fname)[0]
    dst_mask = os.path.join(cloth_mask_dir, stem + ".jpg")
    if precomputed_cloth_masks:
        src_mask = os.path.join(precomputed_cloth_masks, stem + ".jpg")
        if not os.path.exists(src_mask):
            raise FileNotFoundError(f"No precomputed cloth mask at {src_mask}")
        shutil.copy2(src_mask, dst_mask)
        print(f"Copied precomputed cloth mask for {stem}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading U²-Net checkpoint from {mask_ckpt} on {device}")
        net = load_mask_model(mask_ckpt, device)
        mask_tf = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        mask = generate_mask(net, os.path.join(cloth_dir, cloth_fname), mask_tf, device)
        mask.save(dst_mask)
        print(f"Generated cloth mask for {stem}")

    # 4. Remove background from person (noop if disabled)
    print("Removing background from person image...")
    # rgba = remove_background(os.path.join(image_dir, image_fname))
    print("→ background removal done")
    img_rgb = Image.open(os.path.join(image_dir, image_fname)).convert("RGB")
    img_rgb.save(os.path.join(image_dir, image_fname))

    # 5. Human parsing (SCHP or DeepLabV3)
    print("Parsing human segmentation...")
    stem = os.path.splitext(image_fname)[0]
    out_parse = os.path.join(image_parse_dir, stem + ".jpg")
    if parse_dataset in ("lip", "atr", "cihp", "pascal"):
        if precomputed_parse_masks:
            src_parse = os.path.join(precomputed_parse_masks, stem + ".png")
            if not os.path.exists(src_parse):
                raise FileNotFoundError(f"No precomputed parse mask at {src_parse}")
            shutil.copy2(src_parse, out_parse)
            print(f"Copied precomputed parse mask for {stem}")
        else:
            schp_out = os.path.join(out_dir, "schp-parse")
            os.makedirs(schp_out, exist_ok=True)
            cmd = [
                "python3",
                os.path.join(schp_root, "simple_extractor.py"),
                "--dataset",
                parse_dataset,
                "--model-restore",
                parse_ckpt,
                "--input-dir",
                image_dir,
                "--output-dir",
                schp_out,
            ]
            subprocess.run(cmd, check=True)
            shutil.copy2(os.path.join(schp_out, stem + ".png"), out_parse)
    else:
        arr = np.array(img_rgb, dtype=np.uint8)
        mask = parse_human(arr, parse_ckpt, parse_dataset)
        Image.fromarray(mask.astype("uint8")).save(out_parse)

    # 6. Pose estimation (generate or copy)
    print(f"Estimating pose via {pose_backend}...")
    if precomputed_pose_img and precomputed_pose_json:
        stem = os.path.splitext(image_fname)[0]
        # Mediapipe outputs <stem>_pose.png and <stem>_keypoints.json
        img_src = os.path.join(precomputed_pose_img, f"{stem}_keypoints.jpg")
        json_src = os.path.join(precomputed_pose_json, f"{stem}_keypoint.json")

        if not os.path.exists(img_src):
            raise FileNotFoundError(f"Pose overlay not found: {img_src}")
        if not os.path.exists(json_src):
            raise FileNotFoundError(f"Pose JSON not found:     {json_src}")

        shutil.copy2(img_src, os.path.join(openpose_img_dir, os.path.basename(img_src)))
        shutil.copy2(
            json_src, os.path.join(openpose_json_dir, os.path.basename(json_src))
        )
        print(f"Copied pose overlay and JSON for {stem}")
    else:
        if pose_backend == "openpose":
            if not openpose_bin:
                raise ValueError("--openpose-bin is required for openpose backend")
            run_openpose(image_dir, openpose_json_dir, openpose_img_dir, openpose_bin)
        else:
            run_mediapipe(image_dir, openpose_json_dir, openpose_img_dir)

    print(f"Preprocessing complete. All artifacts in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Full preprocessing runner for virtual try-on"
    )
    p.add_argument(
        "--precomputed-cloth-mask-dir",
        help="Directory of precomputed U²-Net cloth masks (.jpg)",
    )
    p.add_argument("--person", "-p", required=True, help="Path to person image")
    p.add_argument("--cloth", "-c", required=True, help="Path to cloth image")
    p.add_argument("--out-dir", "-o", required=True, help="Where to emit outputs")
    p.add_argument("--width", type=int, default=768, help="Resize width")
    p.add_argument("--height", type=int, default=1024, help="Resize height")
    p.add_argument(
        "--resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bilinear",
    )
    p.add_argument("--mask-ckpt", default="checkpoints/cloth_segm_u2net_latest.pth")
    p.add_argument("--parse-ckpt", default="checkpoints/final.pth")
    p.add_argument(
        "--parse-dataset",
        choices=["lip", "atr", "cihp", "pascal", "deeplab"],
        default="lip",
    )
    p.add_argument(
        "--pose-backend", choices=["openpose", "mediapipe"], default="mediapipe"
    )
    p.add_argument("--openpose-bin", help="Path to OpenPose binary if using openpose")
    p.add_argument("--schp-root", default=default_schp_root(), help="Path to SCHP repo")
    p.add_argument(
        "--precomputed-parse-masks",
        "--precomputed-masks",
        dest="precomputed_parse_masks",
        help="Directory of precomputed SCHP human-parsing masks (.jpg)",
    )
    p.add_argument(
        "--precomputed-pose-img", help="Folder of precomputed pose-overlay PNGs"
    )
    p.add_argument(
        "--precomputed-pose-json", help="Folder of precomputed OpenPose JSONs"
    )

    args = p.parse_args()
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
        args.openpose_bin,
        args.schp_root,
        args.precomputed_cloth_mask_dir,
        args.precomputed_parse_masks,
        args.precomputed_pose_img,
        args.precomputed_pose_json,
    )
