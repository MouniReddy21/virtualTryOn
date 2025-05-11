# src/models/tryon_model.py

import torch
from networks import SegGenerator, GMM, ALIASGenerator


class TryOnModel:
    def __init__(self, opt, seg_ckpt, warp_ckpt, comp_ckpt, use_affine=False):
        # 1) Segmentation network
        self.seg = SegGenerator(opt, input_nc=opt.input_nc, output_nc=opt.semantic_nc)
        self.seg.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
        self.seg.eval()

        # 2) Warper (GMM warper from networks.py)
        #    If you want a cheap affine fallback, you could subclass or wrap GMM,
        #    but by default use the full TPS-based warper:
        self.warp = GMM(opt, inputA_nc=opt.input_nc, inputB_nc=opt.input_nc)
        self.warp.load_state_dict(torch.load(warp_ckpt, map_location="cpu"))
        self.warp.eval()

        # 3) Composer (ALIASGenerator)
        self.comp = ALIASGenerator(opt, input_nc=opt.input_nc)
        self.comp.load_state_dict(torch.load(comp_ckpt, map_location="cpu"))
        self.comp.eval()

    @torch.no_grad()
    def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
        # - person_t: [B,3,H,W], cloth_t: [B,3,H,W]
        # - cloth_mask: [B,1,H,W], parse_map: [B,C,H,W], pose_img: [B,3,H,W]
        seg_map = self.seg(torch.cat([person_t, parse_map, pose_img], dim=1))
        theta, grid = self.warp(cloth_t * cloth_mask, person_t)
        output = self.comp(person_t, cloth_t, seg_map, cloth_mask)
        return output
