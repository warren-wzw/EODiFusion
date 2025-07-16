# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch.nn.functional as F

from model.core import build_pixel_sampler
from model.ops import resize
from ..builder import build_loss
from ..losses import accuracy
from ..losses.fusion_loss import Fusionloss

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

        # ----------------- 低分辨率分支处理 -----------------

        # 第一层特征提取 (深度可分离卷积)
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, groups=64),  # 降低通道数
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 第二层特征提取
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 并行路径
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.grouped_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=2),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64+32, 64, 1),  # 降低融合后的通道数
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 高分辨率分支处理 -----------------
        self.high_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 跨分辨率融合模块 -----------------
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(64+64, 128, 3, padding=1),  # 降低卷积输出通道数
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 输出模块 -----------------
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # ----------------- 注意力机制 -----------------
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 8, 1),  # 降低通道数
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        # ================= 低分辨率分支处理 =================
        x_l1 = self.low_conv1(x_low)  # [b,64,80,120]
        x_dilated = self.dilated_conv(x_l1)  # [b,64,80,120]
        x_grouped = self.grouped_conv(self.low_conv2(x_l1))  # [b,32,80,120]

        low_fusion = torch.cat([x_dilated, x_grouped], dim=1)  # [b,96,80,120]
        low_fusion = self.fusion_conv(low_fusion)  # [b,64,80,120]

        low_up = F.interpolate(low_fusion, scale_factor=4, mode='bilinear', align_corners=False)  # [b,64,320,480]

        # ================= 高分辨率分支处理 =================
        high_feat = self.high_conv(x_high)  # [b,64,320,480]

        # ================= 跨分辨率融合 =================
        combined = torch.cat([low_up, high_feat], dim=1)  # [b,128,320,480]
        fused = self.cross_fusion(combined)  # [b,64,320,480]

        # ================= 最终输出 =================
        output = self.output_conv(fused)  # [b,3,320,480]

        # ================= 注意力增强 =================
        se_weight = self.se_block(output)  # [b,3,1,1]
        output = output * se_weight  # [b,3,320,480]

        return output        
    
class FusionModule_(nn.Module):
    def __init__(self):
        super(FusionModule_, self).__init__()
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

class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=256,
                 channels=256,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 dataset_name=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.4),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        elif loss_decode is None:
            pass
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        self.reconstruct = FusionModule()
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:#skip
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:#skip
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,#None
                    ignore_index=self.ignore_index)#self.ignore_index=255
            else:#skip
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss
    
    @force_fp32(apply_to=('fusion_out', ))
    def losses_fusion(self, vi,ir,fusion_out):
        """Compute segmentation loss."""
        alpha=1.0
        loss_func = Fusionloss().to(vi.device)
        loss_in, loss_grad = loss_func(image_vis=vi, image_ir=ir,  generate_img=fusion_out)
        loss_fs = loss_in + alpha * loss_grad
        
        return loss_fs
