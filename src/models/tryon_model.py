# src/models/tryon_model.py

import torch
from .networks import SegGenerator, GMM, ALIASGenerator
from src.utils.utils import gen_noise, load_checkpoint, save_images
import torch.nn.functional as F
import os
from PIL import Image as PILImage
import numpy as np


class TryOnModel:
    def __init__(self, opt, seg_ckpt, warp_ckpt, comp_ckpt, use_affine=False):
        # 1) Segmentation network
        self.seg = SegGenerator(opt, input_nc=opt.input_nc, output_nc=opt.semantic_nc)
        self.seg.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
        self.seg.eval()

        # 2) Warper (GMM warper from networks.py)
        #    If you want a cheap affine fallback, you could subclass or wrap GMM,
        #    but by default use the full TPS-based warper:
        # self.warp = GMM(opt, inputA_nc=opt.input_nc, inputB_nc=opt.input_nc)
        self.warp = GMM(opt, inputA_nc=7, inputB_nc=3)
        self.warp.load_state_dict(torch.load(warp_ckpt, map_location="cpu"))
        self.warp.eval()

        # 3) Composer (ALIASGenerator)
        #    The checkpoint was trained with semantic_nc=7 (7 parse channels)
        #    + 1 misalign mask = 8 label_nc, and input_nc=3 person +3 pose +3 warped cloth = 9.
        orig_sem = opt.semantic_nc
        opt.semantic_nc = 7
        self.comp = ALIASGenerator(opt, input_nc=9)
        self.comp.load_state_dict(torch.load(comp_ckpt, map_location="cpu"))
        self.comp.eval()
        opt.semantic_nc = orig_sem

    @torch.no_grad()
    def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
        # parse_map here is your full SCHP parse; we want to zero out
        # the cloth+arms channels to match the paper’s Sₐ step:

        # garment_idx, left_arm_idx, right_arm_idx = 3, 5, 6
        # s_a = parse_map.clone()
        # for i in (garment_idx, left_arm_idx, right_arm_idx):
        #     s_a[:, i, :, :] = 0
        # parse_map = s_a

        # 1) Segmentation (full 13-way)
        seg_input = torch.cat(
            [
                cloth_mask,
                cloth_t * cloth_mask,
                parse_map,
                pose_img,
                gen_noise(cloth_mask.shape).to(cloth_mask),
            ],
            dim=1,
        )
        parse_pred13 = self.seg(seg_input)  # (1,13,H,W)

        # ─── DEBUG: dump all 13 semantic channels ──────────────────────────────────

        os.makedirs("debug_semantics", exist_ok=True)
        for i in range(parse_pred13.shape[1]):
            # threshold at 0.5 to get a binary mask
            mask_i = (parse_pred13[0, i].cpu().numpy() > 0.5).astype(np.uint8) * 255
            PILImage.fromarray(mask_i).save(f"debug_semantics/sem{i:02d}.png")
        print("→ dumped debug_semantics/sem00.png … sem12.png")
        # ─────────────────────────────────────────────────────────────────────────

        # 2) Keep only the first 7 channels for the ALIAS composer
        #    (it was trained on 7 semantic classes + 1 misalign mask)
        parse_pred = parse_pred13[:, :7, :, :]  # (1,7,H,W)

        # ——— DEBUG: dump each of the 7 semantic channels to disk ———
        debug_dir = "debug_parse"
        os.makedirs(debug_dir, exist_ok=True)
        for i in range(parse_pred.shape[1]):
            channel = parse_pred[0, i].cpu().numpy()  # H×W in [0,1]
            mask = (channel * 255).astype("uint8")  # scale to [0,255]
            PILImage.fromarray(mask).save(f"{debug_dir}/parse_ch{i}.png")
        print(
            f"[debug] saved parse channels to {debug_dir}/parse_ch0.png … parse_ch{parse_pred.shape[1]-1}.png"
        )
        # ——————————————————————————————————————————————————————————————

        # 3) Warp
        # parse_cloth = parse_pred[:, 2:3]  # the "cloth" channel
        # 3) Warp (now use channel 1 as the garment mask)
        cloth_chan_idx = 4
        parse_cloth = parse_pred[:, cloth_chan_idx : cloth_chan_idx + 1]  # (1,1,H,W)

        gmm_input = torch.cat([parse_cloth, pose_img, person_t], dim=1)  # (1,7,H,W)
        _, grid = self.warp(gmm_input, cloth_t)
        # warped_c = F.grid_sample(cloth_t, grid, padding_mode="border")
        warped_c = F.grid_sample(
            cloth_t, grid, padding_mode="border", align_corners=True
        )
        os.makedirs("debug_warp", exist_ok=True)
        wiz = (warped_c.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        PILImage.fromarray((wiz.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            "debug_warp/warped_cloth.png"
        )

        # 4) Compose
        # misalign_mask = (parse_pred[:, 2:3] - cloth_mask).clamp(min=0)  # (1,1,H,W)
        misalign_mask = (parse_cloth - cloth_mask).clamp(min=0)

        os.makedirs("debug_mask", exist_ok=True)
        PILImage.fromarray(
            (misalign_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        ).save("debug_mask/misalign.png")

        parse_div = torch.cat([parse_pred, misalign_mask], dim=1)  # (1,8,H,W)
        # parse_div[:, 2:3] -= misalign_mask  # adjust cloth channel
        parse_div[:, cloth_chan_idx : cloth_chan_idx + 1] -= misalign_mask

        # ** THIS is the 9-channel input to ALIAS **
        x_input = torch.cat([person_t, pose_img, warped_c], dim=1)

        out = self.comp(x_input, parse_pred, parse_div, misalign_mask)
        return out

    @torch.no_grad()
    def to(self, device):
        self.seg.to(device)
        self.warp.to(device)
        self.comp.to(device)
        return self

    def eval(self):
        self.seg.eval()
        self.warp.eval()
        self.comp.eval()
        return self
