# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    修复维度问题的PyTorch实现
    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: 标量偏移值
    Returns:
        mask: shape [D, D, D], torch.int32 表面掩码
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)
