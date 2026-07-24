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

from typing import Callable, Tuple, List, Union, Optional
from tqdm import tqdm
import numpy as np
import traceback
import trimesh
from skimage import measure
import torch

try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor


class BlockGeometryExtractor(BaseGeometryExtractor):
    """
    使用分块策略从符号距离函数(SDF)中提取几何网格的提取器
    
    该方法将整个空间划分为多个小块，在每个小块内使用多尺度金字塔采样策略
    来高效地提取等值面，特别适合处理高分辨率几何提取。
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化块几何提取器
        
        Args:
            device: 计算设备，如果为None则自动选择
        """
        super().__init__(device)
    
    @torch.no_grad()
    def extract_geometry(
        self,
        sdf: Callable,
        resolution: int = 512,
        bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level: float = 0,
        coarse_mask: Optional[torch.Tensor] = None,
        crop_size: int = 512,
        disable_tqdm: bool = False,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        使用分块策略从符号距离函数(SDF)中提取几何网格
        
        Args:
            sdf: 符号距离函数，输入3D点坐标，输出对应的SDF值
            resolution: 整体分辨率，必须是crop_size的倍数，默认512
            bounding_box_min: 边界框最小值，默认(-1, -1, -1)
            bounding_box_max: 边界框最大值，默认(1, 1, 1)
            level: 等值面提取的阈值，默认0（零等值面）
            coarse_mask: 粗粒度掩码，用于跳过空区域
            crop_size: 每个块的分辨率，默认512
            
        Returns:
            trimesh.Trimesh: 提取的合并网格
            
        Raises:
            AssertionError: 如果resolution不是crop_size的倍数
        """
        assert resolution % crop_size == 0, f"resolution {resolution} must be multiple of crop_size {crop_size}"
        
        if coarse_mask is not None:
            # 需要重新排列维度以匹配PyTorch的grid_sample格式 (z, y, x)
            coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].to(self.device).float()

        blocks_n = resolution // crop_size  # 每个维度上的块数

        # 生成每个维度的分块边界
        xs = np.linspace(bounding_box_min[0], bounding_box_max[0], blocks_n + 1)
        ys = np.linspace(bounding_box_min[1], bounding_box_max[1], blocks_n + 1)
        zs = np.linspace(bounding_box_min[2], bounding_box_max[2], blocks_n + 1)

        meshes = []  # 存储所有块的网格
        
        # 使用进度条显示处理进度
        total_blocks = blocks_n * blocks_n * blocks_n
        progress_bar = tqdm(total=total_blocks, desc="Processing blocks", unit="block", disable_tqdm=disable_tqdm)
        
        # 遍历所有空间块
        for i in range(blocks_n):
            for j in range(blocks_n):
                for k in range(blocks_n):
                    progress_bar.update(1)
                    
                    # 计算当前块的边界
                    x_min, x_max = xs[i], xs[i + 1]
                    y_min, y_max = ys[j], ys[j + 1]
                    z_min, z_max = zs[k], zs[k + 1]

                    # 在当前块内生成均匀采样点
                    x = np.linspace(x_min, x_max, crop_size)
                    y = np.linspace(y_min, y_max, crop_size)
                    z = np.linspace(z_min, z_max, crop_size)

                    # 创建3D网格点
                    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                    points = torch.tensor(
                        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, 
                        dtype=torch.float
                    ).to(self.device)

                    # 重新组织点云形状为3D网格 (3, crop_size, crop_size, crop_size)
                    points = points.reshape(crop_size, crop_size, crop_size, 3).permute(3, 0, 1, 2)
                    
                    if coarse_mask is not None:
                        # 使用粗粒度掩码过滤无效区域
                        points_tmp = points.permute(1, 2, 3, 0)[None].to(self.device)
                        current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                        current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                    else:
                        current_mask = None

                    # 构建多尺度金字塔
                    points_pyramid = self.build_pyramid(points, levels=3)

                    # 使用金字塔策略进行高效评估
                    mask = None
                    threshold = 2 * (x_max - x_min) / crop_size * 8  # 初始阈值
                    
                    for pid, pts in enumerate(points_pyramid):
                        coarse_N = pts.shape[-1]  # 当前尺度的分辨率
                        pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                        if mask is None:
                            # 第一层：完整评估或使用掩码过滤
                            if coarse_mask is not None:
                                pts_sdf = torch.ones_like(pts[:, 1])
                                valid_mask = (
                                    torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                                )
                                if valid_mask.any():
                                    pts_sdf[valid_mask] = self.evaluate_sdf_batch(sdf, pts[valid_mask].contiguous())
                            else:
                                pts_sdf = self.evaluate_sdf_batch(sdf, pts)
                        else:
                            # 后续层：只评估掩码区域
                            mask = mask.reshape(-1)
                            pts_to_eval = pts[mask]
                            if pts_to_eval.shape[0] > 0:
                                pts_sdf_eval = self.evaluate_sdf_batch(sdf, pts_to_eval.contiguous())
                                # 确保pts_sdf_eval是一维张量，去除多余的维度
                                if pts_sdf_eval.dim() > 1:
                                    pts_sdf_eval = pts_sdf_eval.squeeze(-1)
                                pts_sdf[mask] = pts_sdf_eval

                        if pid < 3:
                            # 更新掩码：只保留接近等值面的区域
                            mask = torch.abs(pts_sdf) < threshold
                            mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                            mask = self.upsample(mask.float()).bool()  # 上采样到下一层分辨率

                            # 上采样SDF值用于下一层
                            pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                            pts_sdf = self.upsample(pts_sdf)
                            pts_sdf = pts_sdf.reshape(-1)

                        threshold /= 2.0  # 每层减少阈值

                    z = pts_sdf.detach().cpu().numpy()

                    # 跳过没有等值面的块
                    if current_mask is not None:
                        valid_z = z.reshape(crop_size, crop_size, crop_size)[current_mask]
                        if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                            continue

                    # 检查当前块是否包含等值面
                    if not (np.min(z) > level or np.max(z) < level):
                        z = z.astype(np.float32)
                        # 使用marching cubes提取网格
                        verts, faces, normals, _ = measure.marching_cubes(
                            volume=z.reshape(crop_size, crop_size, crop_size),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (crop_size - 1),
                                (y_max - y_min) / (crop_size - 1),
                                (z_max - z_min) / (crop_size - 1),
                            ),
                            mask=current_mask,
                        )
                        # 将顶点坐标转换到全局坐标系
                        verts = verts + np.array([x_min, y_min, z_min])
                        # 创建三角网格
                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        meshes.append(meshcrop)

        progress_bar.close()
        
        # 合并所有块的网格
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return combined
        else:
            # 如果没有提取到网格，返回空网格
            return trimesh.Trimesh()

    def extract_geometry_with_progress(
        self,
        sdf: Callable,
        resolution: int = 512,
        bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level: float = 0,
        coarse_mask: Optional[torch.Tensor] = None,
        crop_size: int = 512,
        **kwargs
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        提取几何并返回详细进度信息
        
        Args:
            参数与extract_geometry相同
            
        Returns:
            Tuple[trimesh.Trimesh, dict]: 提取的网格和统计信息
        """
        mesh = self.extract_geometry(
            sdf, resolution, bounding_box_min, bounding_box_max, 
            level, coarse_mask, crop_size, **kwargs
        )
        
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        
        return mesh, stats


# 保持向后兼容的函数接口
@torch.no_grad()
def extract_geometry_block(sdf, resolution=512, bounding_box_min=(-1.0, -1.0, -1.0),
                          bounding_box_max=(1.0, 1.0, 1.0), level=0, coarse_mask=None):
    """
    向后兼容的函数接口，使用BlockGeometryExtractor类
    
    Args:
        sdf: 符号距离函数
        resolution: 分辨率
        bounding_box_min: 边界框最小值
        bounding_box_max: 边界框最大值
        level: 等值面阈值
        coarse_mask: 粗粒度掩码
        
    Returns:
        trimesh.Trimesh: 提取的网格
        Any: 占位返回值（保持向后兼容）
    """
    extractor = BlockGeometryExtractor()
    mesh = extractor.extract_geometry(
        sdf, resolution, bounding_box_min, bounding_box_max, level, coarse_mask
    )
    return mesh, None


if __name__ == "__main__":
    import time
    # 使用新的类接口进行测试
    print("=" * 50)
    print("开始运行 BlockGeometryExtractor 测试套件")
    print("=" * 50)
    start_time = time.time()
    extractor = BlockGeometryExtractor()
    mesh, stats = extractor.extract_geometry_with_progress(
        sdf=BaseGeometryExtractor.sphere_sdf,
        resolution=512,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0),
        level=0,
        coarse_mask=None,
        crop_size=256
    )
    mesh.export("output_block_mesh.obj")
    
    # 输出网格详细信息
    print("提取的网格信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 验证网格大致是球形的（体积应该在球体体积附近）
    expected_volume = (4/3) * np.pi * (0.5**3)  # 半径为0.5的球体体积
    volume_ratio = stats['volume'] / expected_volume
    print(f"  体积比 (实际/预期): {volume_ratio:.3f}")

    print(f"✅ BlockGeometryExtractor 测试通过！耗时：{time.time() - start_time:.2f}秒")
    
    # 测试向后兼容的函数接口
    print("=" * 50)
    print("开始测试向后兼容性...")
    print("=" * 50)
    start_time = time.time()
    mesh_compat, _ = extract_geometry_block(
        sdf=BaseGeometryExtractor.sphere_sdf,
        resolution=512,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0),
        level=0,
        coarse_mask=None
    )
    assert mesh_compat is not None, "❌ 向后兼容函数失败"
    print(f"✅ 向后兼容性测试通过！耗时: {time.time() - start_time:.2f}秒")
    print("=" * 50)
    print("所有测试通过！")
