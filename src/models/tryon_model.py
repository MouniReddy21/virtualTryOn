# # src/models/tryon_model.py

# import torch
# from .networks import SegGenerator, GMM, ALIASGenerator
# from src.utils.utils import gen_noise, load_checkpoint, save_images


# class TryOnModel:
#     def __init__(self, opt, seg_ckpt, warp_ckpt, comp_ckpt, use_affine=False):
#         # 1) Segmentation network
#         self.seg = SegGenerator(opt, input_nc=opt.input_nc, output_nc=opt.semantic_nc)
#         self.seg.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
#         self.seg.eval()

#         # 2) Warper (GMM warper from networks.py)
#         #    If you want a cheap affine fallback, you could subclass or wrap GMM,
#         #    but by default use the full TPS-based warper:
#         # self.warp = GMM(opt, inputA_nc=opt.input_nc, inputB_nc=opt.input_nc)
#         self.warp = GMM(opt, inputA_nc=7, inputB_nc=3)
#         self.warp.load_state_dict(torch.load(warp_ckpt, map_location="cpu"))
#         self.warp.eval()

#         # 3) Composer (ALIASGenerator)
#         # self.comp = ALIASGenerator(opt, input_nc=opt.input_nc)
#         # self.comp.load_state_dict(torch.load(comp_ckpt, map_location="cpu"))
#         # self.comp.eval()
#         # orig_sem_nc = opt.semantic_nc
#         # opt.semantic_nc = 7
#         # self.comp = ALIASGenerator(opt, input_nc=9)
#         # # load_ckpt(self.comp, comp_ckpt)
#         # self.comp.eval()
#         # opt.semantic_nc = orig_sem_nc
#         # ALIASGenerator’s `x` input is the person image: 3 channels
#         # 3) Composer (ALIASGenerator)
#         #    The checkpoint was trained with semantic_nc=7 (7 parse channels)
#         #    + 1 misalign mask = 8 label_nc, and input_nc=3 person +3 pose +3 warped cloth = 9.
#         orig_sem = opt.semantic_nc
#         opt.semantic_nc = 7
#         self.comp = ALIASGenerator(opt, input_nc=9)
#         self.comp.load_state_dict(torch.load(comp_ckpt, map_location="cpu"))
#         self.comp.eval()
#         opt.semantic_nc = orig_sem

#     # @torch.no_grad()
#     # def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
#     #     # - person_t: [B,3,H,W], cloth_t: [B,3,H,W]
#     #     # - cloth_mask: [B,1,H,W], parse_map: [B,C,H,W], pose_img: [B,3,H,W]
#     #     c_masked = cloth_t * cloth_mask  # [1,3,H,W]

#     #     noise = gen_noise(cloth_mask.shape).to(cloth_mask)  # [1,1,H,W]
#     #     seg_input = torch.cat([cloth_mask, c_masked, parse_map, pose_img, noise], dim=1)
#     #     parse_pred = self.seg(seg_input)  # [1,13,H,W]

#     #     # now warp & compose as before
#     #     theta, grid = self.warp(cloth_t * cloth_mask, person_t)
#     #     out = self.comp(person_t, cloth_t, parse_pred, cloth_mask)
#     #     return out

#     # @torch.no_grad()
#     # def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
#     #     # 1) segmentation
#     #     seg_input = torch.cat(
#     #         [person_t, parse_map, pose_img], dim=1
#     #     )  # (1,3+13+3=19,H,W)
#     #     # RIGHT: 1 + 3 + 13 + 3 + 1 = 21 channels
#     #     # 1) cloth mask
#     #     # 2) cloth masked (cloth_t * cloth_mask)
#     #     # 3) parse agnostic
#     #     # 4) pose overlay
#     #     # 5) noise
#     #     c_masked = cloth_t * cloth_mask  # 3ch
#     #     noise = gen_noise(cloth_mask.shape).to(cloth_mask)  # 1ch
#     #     seg_input = torch.cat(
#     #         [
#     #             cloth_mask,  # 1 channel
#     #             c_masked,  # 3 channels
#     #             parse_map,  # 13 channels
#     #             pose_img,  # 3 channels
#     #             noise,  # 1 channel
#     #         ],
#     #         dim=1,
#     #     )  # total = 21
#     #     parse_pred = self.seg(seg_input)  # (1,13,H,W)

#     #     # 2) warp
#     #     # extract the “cloth region” from the parse:
#     #     parse_cloth = parse_pred[:, 2:3]  # (1,1,H,W)
#     #     agnostic = person_t  # ideally person with cloth masked out
#     #     # build the 7-channel GMM input: [parse_cloth(1) + pose_img(3) + agnostic(3)]
#     #     gmm_input = torch.cat([parse_cloth, pose_img, agnostic], dim=1)  # (1,7,H,W)
#     #     theta, grid = self.warp(gmm_input, cloth_t)  # inputB=cloth_t
#     #     warped_cloth = torch.nn.functional.grid_sample(
#     #         cloth_t, grid, padding_mode="border"
#     #     )

#     #     # 3) final composition
#     #     out = self.comp(person_t, warped_cloth, parse_pred, cloth_mask)
#     #     return out

#     @torch.no_grad()
#     def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
#         # 1) Segmentation
#         seg_input = torch.cat(
#             [
#                 cloth_mask,
#                 cloth_t * cloth_mask,
#                 parse_map,
#                 pose_img,
#                 gen_noise(cloth_mask.shape).to(cloth_mask),
#             ],
#             dim=1,
#         )
#         parse_pred = self.seg(seg_input)  # (1,13,H,W)

#         # 2) Warp
#         parse_cloth = parse_pred[:, 2:3]
#         gmm_input = torch.cat(
#             [parse_cloth, pose_img, person_t], dim=1
#         )  # (1,1+3+3=7,H,W)
#         _, grid = self.warp(gmm_input, cloth_t)
#         warped_c = F.grid_sample(cloth_t, grid, padding_mode="border")

#         # 3) Compose
#         misalign_mask = (parse_pred[:, 2:3] - cloth_mask).clamp(min=0)
#         parse_div = torch.cat([parse_pred, misalign_mask], dim=1)
#         parse_div[:, 2:3] -= misalign_mask

#         # ** THIS is the 9-channel input to ALIAS **
#         x_input = torch.cat([person_t, pose_img, warped_c], dim=1)
#         out = self.comp(x_input, parse_pred, parse_div, misalign_mask)
#         return out

#     @torch.no_grad()
#     def to(self, device):
#         self.seg.to(device)
#         self.warp.to(device)
#         self.comp.to(device)
#         return self

#     def eval(self):
#         self.seg.eval()
#         self.warp.eval()
#         self.comp.eval()
#         return self

# src/models/tryon_model.py

import torch
from .networks import SegGenerator, GMM, ALIASGenerator
from src.utils.utils import gen_noise, load_checkpoint, save_images
import torch.nn.functional as F


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

    # @torch.no_grad()
    # def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
    #     # - person_t: [B,3,H,W], cloth_t: [B,3,H,W]
    #     # - cloth_mask: [B,1,H,W], parse_map: [B,C,H,W], pose_img: [B,3,H,W]
    #     c_masked = cloth_t * cloth_mask  # [1,3,H,W]

    #     noise = gen_noise(cloth_mask.shape).to(cloth_mask)  # [1,1,H,W]
    #     seg_input = torch.cat([cloth_mask, c_masked, parse_map, pose_img, noise], dim=1)
    #     parse_pred = self.seg(seg_input)  # [1,13,H,W]

    #     # now warp & compose as before
    #     theta, grid = self.warp(cloth_t * cloth_mask, person_t)
    #     out = self.comp(person_t, cloth_t, parse_pred, cloth_mask)
    #     return out

    @torch.no_grad()
    def infer(self, person_t, cloth_t, cloth_mask, parse_map, pose_img):
        # 1) Segmentation
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
        parse_pred = self.seg(seg_input)  # (1,13,H,W)

        # 2) Warp
        parse_cloth = parse_pred[:, 2:3]
        gmm_input = torch.cat(
            [parse_cloth, pose_img, person_t], dim=1
        )  # (1,1+3+3=7,H,W)
        _, grid = self.warp(gmm_input, cloth_t)
        warped_c = F.grid_sample(cloth_t, grid, padding_mode="border")

        # 3) Compose
        misalign_mask = (parse_pred[:, 2:3] - cloth_mask).clamp(min=0)
        parse_div = torch.cat([parse_pred, misalign_mask], dim=1)
        parse_div[:, 2:3] -= misalign_mask

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
