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
         

def reduce_contrast(img_tensor, factor=0.5):
    mean = img_tensor.mean(dim=(2, 3), keepdim=True)
    return torch.clamp(mean + factor * (img_tensor - mean), 0, 1)

def ContrastEnhancer(): 
    print()

def visualize_feature_activations(feature, original_img, ir_img, img_metas, save_dir='./'):
    """可视化特征激活热力图"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from torch.nn import functional as F
    
    os.makedirs(save_dir, exist_ok=True)
    feature_map = feature[0].mean(dim=0)  # [h/4, w/4]

    feature_map = F.interpolate(
        feature_map.unsqueeze(0).unsqueeze(0),  # [1, 1, h/4, w/4]
        size=original_img.shape[2:],  # 原始图像尺寸
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    rgb_img = original_img[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    # 绘制可视化图像
    plt.figure(figsize=(15, 5))
    # 原始RGB图像
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(feature_map, cmap='jet')
    plt.title('Attention Heatmap')
    plt.axis('off')
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_img)
    plt.imshow(feature_map, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    # 保存图像
    filename = os.path.basename(img_metas[0]['filename']).split('.')[0]
    plt.savefig(f"{save_dir}/{filename}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {save_dir}/{filename}_heatmap.png")        
