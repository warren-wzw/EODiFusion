import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import cv2
import random

from model.ops import resize
from torch.special import expm1
from einops import rearrange, repeat

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from .utils import *
from ..losses.fusion_loss import Fusionloss

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    
@SEGMENTORS.register_module()
class EODiFusion(EncoderDecoder):
    
    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 accumulation=False,
                 **kwargs):
        super(EODiFusion, self).__init__(**kwargs)

        self.timesteps = timesteps
        self.randsteps = randsteps
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.log_snr = alpha_cosine_log_snr

        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )
        self.gt_down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )
        self.fusion_loss = Fusionloss()
        self.enhance=ContrastEnhancer()
        self.transform = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times
    
    def create_noise(self,gt_down,times,img_ir):
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img_ir, noise_level)#turn [b]->[b,1,1,1]
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        input_times = self.time_mlp(noise_level)
        return noised_gt,input_times,noise
    
    def ddim_sample(self, feature,img,ir):
        b, c, h, w, device = *feature.shape, feature.device #[b,256,h/4,w/4]
        time_pairs = self._get_sampling_timesteps(b, device=device)
        feature = repeat(feature, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, feature.shape[1], h, w), device=device)
        for _, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([feature,mask_t], dim=1) #[b,768,h/4,w/4]
            feat=self.transform(feat)
            log_snr = self.log_snr(times_now)#[1]
            log_snr_next = self.log_snr(times_next)#[1]
            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr) #pad log_snr [1]-[1,1,1,1]
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next) #pad log_snr [1]-[1,1,1,1]
            sigma, alpha = log_snr_to_alpha_sigma(padded_log_snr)
            sigma_next, alpha_next = log_snr_to_alpha_sigma(padded_log_snr_next)
            input_times = self.time_mlp(log_snr)#1->[1,1024]
            fusion_out,memory = self.decode_head.forward_fusion([feat], input_times,img,ir) # [bs, 256,h/4,w/4 ]-[b,9,h/4,w/4]
            mask_pred = self.gt_down(fusion_out)
            """epsilon_t=(x_t-sigma_t*x_t)/alpha_t"""
            pred_noise = (mask_t - sigma * mask_pred) / alpha.clamp(min=1e-8)
            """x_t-1=alpha_t-1*epsilon_t+sigma_t-1*x_t"""
            mask_t = alpha_next*pred_noise+sigma_next*mask_pred
        return fusion_out
       
    def encode_decode(self, img,ir, img_metas):
        """turn RGB to YCbCr"""
        img, img_Cb, img_Cr = RGB2YCrCb(img)
        """create input"""
        img_ir = torch.cat([img, ir], dim=1)#[b,4,h,w]
        """extract feature"""
        feature = self.extract_feat(img_ir)[0]#[b,256, h/4, w/4]
        visualize_feature_activations(feature, img, ir,img_metas)
        """ddim sample"""
        fusion_out = self.ddim_sample(feature,img,ir)
        """turn YCbCr to RGB"""
        fusion_out = YCbCr2RGB(fusion_out, img_Cb, img_Cr)
        save_single_image(img=fusion_out,save_path_img=img_metas[0]['ori_filename'],
                size=img_metas[0]['ori_shape'][:-1])
        #out = self._auxiliary_head_forward_test([feature], img_metas)
        out = torch.zeros_like(fusion_out)
        return out

    def forward_train(self, img, img_metas, ir,img_ori,ir_ori,gt_semantic_seg):
        losses = dict()
        mask=gt_semantic_seg
        img_ori,ir_ori=img_ori.float(),ir_ori.float()
        """turn RGB to YCbCr"""
        img, _, _ = RGB2YCrCb(img)
        img_ori, _, _ = RGB2YCrCb(img_ori)
        """create input"""
        img_ir = torch.cat([img, ir], dim=1)
        """image"""
        feature = self.extract_feat(img_ir)[0]  # bs, 128, h/4, w/4
        """create pseudo image"""
        fused_image_pseudo = (img_ori+ir_ori)/2
        """gtdown represents the embdding labels """
        gt_down = self.gt_down(fused_image_pseudo)
        """sample time"""
        batch, c, h, w, device, = *feature.shape, feature.device
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],self.sample_range[1])  
        """create noise based on gt_down"""
        noised_gt,input_times,noise=self.create_noise(gt_down,times,img_ir)
        """conditional input"""
        feat = torch.cat([feature,noised_gt], dim=1)
        feat=self.transform(feat)
        fusion_out,_= self.decode_head.forward_fusion([feat], input_times,img_ori,ir_ori)
        loss_fusion=self.fusion_loss(img_ori,ir_ori,fusion_out,mask,fused_image_pseudo)
        losses.update(loss_fusion)
        pred_noise=self.gt_down(fusion_out)
        losses['loss_diff'] = F.mse_loss(pred_noise, noise, reduction='none')
        """"""
        # fused_low = reduce_contrast(fusion_out, random.uniform(0.66,1.0))
        # fusion_aug = self.enhance(fused_low,mask)
        # losses["loss_aug"] = F.l1_loss(fusion_aug, fusion_out)
        """"""
        # loss_aux = self._auxiliary_head_forward_train([feature], img_metas, gt_semantic_seg)
        # losses.update(loss_aux)
        return losses



    