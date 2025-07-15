import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import cv2

from model.ops import resize
from torch.special import expm1
from einops import rearrange, repeat

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from ..losses.fusion_loss import Fusionloss
from ..losses.synergy_loss import SynergyLoss

 
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def save_single_image(img=None, ir=None, save_path_img=None, save_path_ir=None,size=None):
    if img is not None:
        file_name = save_path_img.split('/')[-1]
        file_name="./out/fusion/"+file_name
        img = resize(
            input=img,
            size=size,
            mode='bilinear',
            align_corners=False)
        img_np = img[0].permute(1, 2, 0).cpu().numpy()  # 变换维度
        img_np = (img_np * 255).astype(np.uint8)  # 反归一化到 [0, 255]
        cv2.imwrite(file_name, img_np)  # 保存可见光图像

    if ir is not None:
        file_name = save_path_img.split('/')[-1]
        file_name="./out/fusion/"+file_name
        ir = resize(
            input=ir,
            size=size,
            mode='bilinear',
            align_corners=False)
        ir_np = ir[0].squeeze(0).cpu().numpy()
        ir_np = (ir_np * 255).astype(np.uint8)  # 反归一化到 [0, 255]
        cv2.imwrite(file_name, ir_np)  # 保存红外图像

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out
         
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

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        """low resolution"""
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, groups=128),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        """feature fusion"""
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128+64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        """high resolution [b,4,320,480]"""
        self.high_conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        """cros  resolution fusion"""
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(128+128, 256, 3, padding=1),  # 融合低分辨率上采样特征和高分辨率特征
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        """low resolution """
        x_l1 = self.low_conv1(x_low)  # [b,128,80,120]
        x_dilated = self.dilated_conv(x_l1)  # [b,128,80,120]
        x_grouped = self.grouped_conv(self.low_conv2(x_l1))  # [b,64,80,120]
        """feature fusion"""
        low_fusion = torch.cat([x_dilated, x_grouped], dim=1)  # [b,192,80,120]
        low_fusion = self.fusion_conv(low_fusion)  # [b,128,80,120]
        """upsample"""
        low_feat = F.interpolate(low_fusion, scale_factor=4, mode='bilinear', align_corners=False)  # [b,128,320,480]
        """high resolution"""
        high_feat = self.high_conv(x_high)  # [b,128,320,480]
        combined = torch.cat([low_feat, high_feat], dim=1)  # [b,256,320,480]
        feat = self.cross_fusion(combined)  # [b,128,320,480]
        output = self.output_conv(feat)  # [b,3,320,480]
        """attention augmentation"""
        se_weight = self.se_block(output)  # [b,3,1,1]
        output = output * se_weight  # [b,3,320,480]
        """"""
        return output

class LearnableContrastEnhancer(torch.nn.Module):

    def __init__(self, init_factor=1.2):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(init_factor))  # 可训练对比度因子

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        return torch.clamp(mean + 1.1 * (x - mean), 0, 1) 

def reduce_contrast(img_tensor, factor=0.5):
    mean = img_tensor.mean(dim=(2, 3), keepdim=True)
    return torch.clamp(mean + factor * (img_tensor - mean), 0, 1)
        
@SEGMENTORS.register_module()
class DiFusionSeg(EncoderDecoder):
    
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
        super(DiFusionSeg, self).__init__(**kwargs)

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
        self.fusion_loss=Fusionloss()
        self.syn_loss=SynergyLoss()
        self.gt_down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )
        self.enhance=LearnableContrastEnhancer()
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
        """ddim sample"""
        fusion_out = self.ddim_sample(feature,img,ir)
        #fusion_out=self.enhance(fusion_out)
        """turn YCbCr to RGB"""
        fusion_out = YCbCr2RGB(fusion_out, img_Cb, img_Cr)
        save_single_image(img=fusion_out,save_path_img=img_metas[0]['ori_filename'],
                size=img_metas[0]['ori_shape'][:-1])

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
        loss_aux = self._auxiliary_head_forward_train([feature], img_metas, gt_semantic_seg)
        losses.update(loss_aux)
        return losses



    