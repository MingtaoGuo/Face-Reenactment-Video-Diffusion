import torch
import torch.nn as nn 
import torch.nn.functional as F
import einops 
from scipy.spatial import ConvexHull # pylint: disable=E0401,E0611
from typing import Union
import numpy as np 

import tyro
from src.liveportrait.config.argument_config import ArgumentConfig
from src.liveportrait.config.inference_config import InferenceConfig
from src.liveportrait.utils.camera import get_rotation_matrix
from src.liveportrait.live_portrait_wrapper import LivePortraitWrapper

def tensor_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """transform torch.Tensor into numpy.ndarray"""
    if isinstance(data, torch.Tensor):
        return data.data.cpu().numpy()
    return data

def calc_motion_multiplier(
    kp_source: Union[np.ndarray, torch.Tensor],
    kp_driving_initial: Union[np.ndarray, torch.Tensor]
) -> float:
    """calculate motion_multiplier based on the source image and the first driving frame"""
    kp_source_np = tensor_to_numpy(kp_source)
    kp_driving_initial_np = tensor_to_numpy(kp_driving_initial)

    source_area = ConvexHull(kp_source_np.squeeze(0)).volume
    driving_area = ConvexHull(kp_driving_initial_np.squeeze(0)).volume
    motion_multiplier = np.sqrt(source_area) / np.sqrt(driving_area)
    # motion_multiplier = np.cbrt(source_area) / np.cbrt(driving_area)

    return motion_multiplier

def to_image(I_s, I_d, res):
    print(res.shape)
    res = F.interpolate(res[0], [256, 256])

    src = []
    drv = []
    rrr = []
    for i in range(16):
        src.append(I_s[i].permute(1, 2, 0).detach().cpu().numpy()*255)
        drv.append(I_d[i].permute(1, 2, 0).detach().cpu().numpy()*255)
        rrr.append(res[i].permute(1, 2, 0).detach().cpu().numpy()*255)
    import numpy as np 
    src = np.concatenate(src, axis=1) 
    drv = np.concatenate(drv, axis=1) 
    rrr = np.concatenate(rrr, axis=1) 
    return np.concatenate([src, drv, rrr], axis=0)

class MotionExtractor(nn.Module):
    def __init__(self, motion_extractor_path):
        super().__init__()
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
        inference_cfg.checkpoint_F = motion_extractor_path + "/appearance_feature_extractor.pth"
        inference_cfg.checkpoint_M = motion_extractor_path + "/motion_extractor.pth"
        inference_cfg.checkpoint_G = motion_extractor_path + "/spade_generator.pth"
        inference_cfg.checkpoint_W = motion_extractor_path + "/warping_module.pth"
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg) 
        self.dtype=torch.float32
    
    def forward(self, I_s, I_d):
        # I_s: [B, F, C, H, W], I_d: [B, F, C, H, W]
        f = I_d.shape[1]

        I_s = (I_s + 1.0) / 2.0 
        I_d = (I_d + 1.0) / 2.0 
        I_s = einops.rearrange(I_s, "b f c h w -> (b f) c h w")
        I_d = einops.rearrange(I_d, "b f c h w -> (b f) c h w")

        I_s = F.interpolate(I_s, [256, 256], mode="bilinear") 
        I_d = F.interpolate(I_d, [256, 256], mode="bilinear")
        with torch.no_grad():
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']  # canonical kp
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])

            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            x_d_info = self.live_portrait_wrapper.get_kp_info(I_d)
            R_d = get_rotation_matrix(x_d_info['pitch'], x_d_info['yaw'], x_d_info['roll'])

            t_d = x_d_info['t']
            exp_d = x_d_info['exp']
            scale_d = x_d_info['scale']

            x_d_new = scale_d[..., None] * (x_c_s @ R_d + exp_d) + t_d[:, None, :]

            ret_dct = self.live_portrait_wrapper.warping_module(f_s, kp_source=x_s, kp_driving=x_d_new)
            # decode
            out = self.live_portrait_wrapper.spade_generator(feature=ret_dct['out'])
            # out = ret_dct['out']
            out = einops.rearrange(out, "(b f) c h w -> b f c h w", f=f)

        return out



class MotionExtractorEval(nn.Module):
    def __init__(self, motion_extractor_path):
        super().__init__()
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
        inference_cfg.checkpoint_F = motion_extractor_path + "/appearance_feature_extractor.pth"
        inference_cfg.checkpoint_M = motion_extractor_path + "/motion_extractor.pth"
        inference_cfg.checkpoint_G = motion_extractor_path + "/spade_generator.pth"
        inference_cfg.checkpoint_W = motion_extractor_path + "/warping_module.pth"
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg) 
        self.dtype=torch.float32
    
    def forward(self, I_src, I_drv):
        # I_src: [B, F, C, H, W], I_drv: [B, F, C, H, W]
        f = I_drv.shape[1]
  
        I_src = (I_src + 1.0) / 2.0 
        I_drv = (I_drv + 1.0) / 2.0 

        I_s = F.interpolate(I_src[:, 0], [256, 256], mode="bilinear") 
        with torch.no_grad():
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']  # canonical kp
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])

            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        I_p_lst = []
        for i in range(f):
            with torch.no_grad():
                I_d_i = F.interpolate(I_drv[:, i], [256, 256], mode="bilinear") 
                x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
                R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
                if i == 0:
                    R_d_0 = R_d_i
                    x_d_0_info = x_d_i_info

                if True:
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                    scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                    t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])

                t_new[..., 2].fill_(0) # zero tz
                x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

                ret_dct = self.live_portrait_wrapper.warping_module(f_s, kp_source=x_s, kp_driving=x_d_i_new)
                I_p_i = self.live_portrait_wrapper.spade_generator(feature=ret_dct['out'])
                I_p_lst.append(I_p_i[:, None])

        out = torch.cat(I_p_lst, dim=1)

        return out

       

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

