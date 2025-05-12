# # src/inference/run_single.py
# import os, json, argparse
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms
# from src.models.tryon_model import TryOnModel


# def load_image(path, mode="RGB"):
#     return Image.open(path).convert(mode)


# def image_to_tensor(img, normalize=True):
#     t = transforms.ToTensor()(img)
#     if normalize:
#         # assume networks were trained on [-1,1] range
#         t = (t - 0.5) * 2.0
#     return t.unsqueeze(0)  # add batch dim


# def load_pose_render(path, size):
#     # load the pose overlay as a float image in [0,1]
#     img = load_image(path, "RGB").resize(size, Image.BILINEAR)
#     return image_to_tensor(img, normalize=False)


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument(
#         "--inputs",
#         required=True,
#         help="output folder from runner.py (contains image/, cloth/, etc.)",
#     )
#     p.add_argument(
#         "--checkpoints",
#         required=True,
#         help="where seg_final.pth, gmm_final.pth, alias_final.pth live",
#     )
#     p.add_argument("--out", required=True, help="where to save the final try-on PNG")
#     p.add_argument("--width", type=int, default=768)
#     p.add_argument("--height", type=int, default=1024)
#     args = p.parse_args()

#     inp = args.inputs
#     W, H = args.width, args.height

#     # 1) load person & cloth
#     person_p = os.path.join(inp, "image", os.listdir(os.path.join(inp, "image"))[0])
#     cloth_p = os.path.join(inp, "cloth", os.listdir(os.path.join(inp, "cloth"))[0])
#     person_img = load_image(person_p, "RGB").resize((W, H), Image.BILINEAR)
#     cloth_img = load_image(cloth_p, "RGB").resize((W, H), Image.BILINEAR)

#     # 2) cloth mask
#     mask_p = os.path.join(
#         inp, "cloth-mask", os.listdir(os.path.join(inp, "cloth-mask"))[0]
#     )
#     cloth_mask = load_image(mask_p, "L").resize((W, H), Image.NEAREST)

#     # 3) human parse
#     parse_p = os.path.join(
#         inp, "image-parse", os.listdir(os.path.join(inp, "image-parse"))[0]
#     )
#     # parse_map = load_image(parse_p, "L").resize((W, H), Image.NEAREST)

#     # # 4) pose render & (optionally) keypoints
#     pose_img_p = os.path.join(
#         inp, "openpose-img", os.listdir(os.path.join(inp, "openpose-img"))[0]
#     )
#     # pose_json_p = os.path.join(
#     #     inp, "openpose-json", os.listdir(os.path.join(inp, "openpose-json"))[0]
#     # )
#     # parse_img = load_image(parse_p, "L").resize((W, H), Image.NEAREST)
#     # parse_np = np.array(parse_img, dtype=np.int64)  # H×W, values in [0..12]
#     # # build one-hot, shape (13,H,W)
#     # parse_oh = np.stack([(parse_np == c).astype(np.uint8) for c in range(13)], axis=0)

#     # # turn everything into torch tensors
#     # # person_t = image_to_tensor(person_img)  # [1,3,H,W] in [-1,1]
#     # parse_t = torch.from_numpy(parse_oh).float().unsqueeze(0)  # [1,13,H,W]

#     # cloth_t = image_to_tensor(cloth_img)
#     # cloth_m_t = image_to_tensor(cloth_mask, normalize=False)  # [1,1,H,W] in {0,1}
#     # parse_t = image_to_tensor(parse_map, normalize=False)  # [1,1,H,W] in {0,1}
#     # pose_render = load_pose_render(pose_img_p, (W, H))  # [1,3,H,W] in [0,1]
#     # 3) human parse → one-hot into 13 channels
#     parse_img = load_image(parse_p, "L").resize((W, H), Image.NEAREST)
#     parse_np = np.array(parse_img, dtype=np.int64)  # H×W, values 0..12
#     parse_oh = np.stack(
#         [(parse_np == c).astype(np.uint8) for c in range(13)], axis=0
#     )  # (13,H,W)

#     # 4) pose render & (optionally) keypoints
#     pose_render = load_pose_render(pose_img_p, (W, H))  # [1,3,H,W] in [0,1]

#     # 5) to torch tensors
#     person_t = image_to_tensor(person_img)  # [1,3,H,W] in [-1,1]
#     cloth_t = image_to_tensor(cloth_img)  # [1,3,H,W]
#     cloth_m_t = image_to_tensor(cloth_mask, normalize=False)  # [1,1,H,W]
#     parse_t = torch.from_numpy(parse_oh).float().unsqueeze(0)  # [1,13,H,W]

#     # 5) run the try-on model
#     # model = TryOnModel(
#     #     os.path.join(args.checkpoints, "seg_final.pth"),
#     #     os.path.join(args.checkpoints, "gmm_final.pth"),
#     #     os.path.join(args.checkpoints, "alias_final.pth"),
#     #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     # )
#     # model.eval()
#     # build a “opt” just like you did in training: semantic_nc=13, plus 8 extra channels
#     from types import SimpleNamespace

#     opt = SimpleNamespace(
#         input_nc=13 + 8,  # 21 channels for SegGenerator
#         semantic_nc=13,
#         load_height=H,
#         load_width=W,
#         init_type="none",
#         init_variance=0.02,
#         grid_size=5,
#         norm_G="spectralaliasinstance",
#         ngf=64,
#         num_upsampling_layers="most",
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TryOnModel(
#         opt,
#         os.path.join(args.checkpoints, "seg_final.pth"),
#         os.path.join(args.checkpoints, "gmm_final.pth"),
#         os.path.join(args.checkpoints, "alias_final.pth"),
#     )
#     # model.to(device).eval()
#     model.seg.to(device).eval()
#     model.warp.to(device).eval()
#     model.comp.to(device).eval()
#     with torch.no_grad():
#         out = model.infer(person_t, cloth_t, cloth_m_t, parse_t, pose_render)
#         # out is [1,3,H,W] in [-1,1] (TanH)
#         out_img = (out.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)  # back to [0,1]
#         to_pil = transforms.ToPILImage()
#         to_pil(out_img).save(args.out)

#     print(f"Saved try-on result to {args.out}")


# if __name__ == "__main__":
#     main()


# src/inference/run_single.py
import os, argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from src.models.tryon_model import TryOnModel


def load_image(path, mode="RGB"):
    return Image.open(path).convert(mode)


def image_to_tensor(img, normalize=True):
    t = transforms.ToTensor()(img)
    if normalize:
        # assume networks were trained on [-1,1] range
        t = (t - 0.5) * 2.0
    return t.unsqueeze(0)  # add batch dim


def load_pose_render(path, size):
    img = load_image(path, "RGB").resize(size, Image.BILINEAR)
    return image_to_tensor(img, normalize=False)


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
        help="where seg_final.pth, gmm_final.pth, alias_final.pth live",
    )
    p.add_argument("--out", required=True, help="where to save the final try-on PNG")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    args = p.parse_args()

    inp = args.inputs
    W, H = args.width, args.height

    # 1) load person & cloth
    person_p = os.path.join(inp, "image", os.listdir(os.path.join(inp, "image"))[0])
    cloth_p = os.path.join(inp, "cloth", os.listdir(os.path.join(inp, "cloth"))[0])
    person_img = load_image(person_p).resize((W, H), Image.BILINEAR)
    cloth_img = load_image(cloth_p).resize((W, H), Image.BILINEAR)

    # 2) cloth mask
    mask_p = os.path.join(
        inp, "cloth-mask", os.listdir(os.path.join(inp, "cloth-mask"))[0]
    )
    cloth_mask = load_image(mask_p, "L").resize((W, H), Image.NEAREST)

    # 3) human parse → one-hot (13 classes)
    parse_p = os.path.join(
        inp, "image-parse", os.listdir(os.path.join(inp, "image-parse"))[0]
    )
    parse_img = load_image(parse_p, "L").resize((W, H), Image.NEAREST)
    parse_np = np.array(parse_img, dtype=np.int64)  # H×W, values 0..12
    parse_oh = np.stack(
        [(parse_np == c).astype(np.uint8) for c in range(13)], axis=0
    )  # (13,H,W)

    # 4) pose render
    pose_img_p = os.path.join(
        inp, "openpose-img", os.listdir(os.path.join(inp, "openpose-img"))[0]
    )
    pose_render = load_pose_render(pose_img_p, (W, H))  # [1,3,H,W]

    # 5) to torch tensors
    person_t = image_to_tensor(person_img)  # [1,3,H,W] in [-1,1]
    cloth_t = image_to_tensor(cloth_img)  # [1,3,H,W]
    cloth_m_t = image_to_tensor(cloth_mask, normalize=False)  # [1,1,H,W]
    parse_t = torch.from_numpy(parse_oh).float().unsqueeze(0)  # [1,13,H,W]

    # 6) build opt and load model
    from types import SimpleNamespace

    opt = SimpleNamespace(
        input_nc=13 + 8,  # SegGenerator expects 13 human classes + 8 extra channels
        semantic_nc=13,
        load_height=192,  # H
        load_width=256,  # W
        init_type="none",
        init_variance=0.02,
        grid_size=5,
        norm_G="spectralaliasinstance",
        ngf=64,
        num_upsampling_layers="most",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TryOnModel(
        opt,
        os.path.join(args.checkpoints, "seg_final.pth"),
        os.path.join(args.checkpoints, "gmm_final.pth"),
        os.path.join(args.checkpoints, "alias_final.pth"),
    )
    # move model components to device
    model.seg.to(device).eval()
    model.warp.to(device).eval()
    model.comp.to(device).eval()

    # ——— NEW: move inputs to same device ———
    # person_t = person_t.to(device)
    # cloth_t = cloth_t.to(device)
    # cloth_m_t = cloth_m_t.to(device)
    # parse_t = parse_t.to(device)
    # pose_render = pose_render.to(device)
    # right after loading `person_img`, `cloth_img`, `parse_img`, `pose_img_p`…

    person_resized = person_img.resize((256, 192), Image.BILINEAR)
    cloth_resized = cloth_img.resize((256, 192), Image.BILINEAR)
    parse_resized = parse_img.resize((256, 192), Image.NEAREST)
    pose_resized = load_image(pose_img_p, "RGB").resize((256, 192), Image.BILINEAR)

    # now turn *those* into tensors:
    person_t = image_to_tensor(person_resized)
    cloth_t = image_to_tensor(cloth_resized)
    cloth_m_t = image_to_tensor(
        cloth_mask.resize((256, 192), Image.NEAREST), normalize=False
    )
    parse_t = (
        torch.from_numpy(
            np.stack(
                [(np.array(parse_resized) == c).astype(np.uint8) for c in range(13)],
                axis=0,
            )
        )
        .float()
        .unsqueeze(0)
    )
    pose_render = image_to_tensor(pose_resized, normalize=False)

    # 7) run inference
    with torch.no_grad():
        out = model.infer(person_t, cloth_t, cloth_m_t, parse_t, pose_render)
        # out: [1,3,H,W] in [-1,1]
        out_img = (out.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)  # → [0,1]
        to_pil = transforms.ToPILImage()
        to_pil(out_img).save(args.out)

    print(f"Saved try-on result to {args.out}")


if __name__ == "__main__":
    main()
