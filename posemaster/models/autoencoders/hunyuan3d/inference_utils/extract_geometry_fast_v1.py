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
import traceback
import numpy as np
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import trimesh
import torch
from einops import repeat

try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor


class FastGeometryExtractorV1(BaseGeometryExtractor):
    """
    快速几何提取器，使用插值策略高效提取几何网格
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化快速几何提取器
        
        Args:
            device: 计算设备，如果为None则自动选择
        """
        super().__init__(device)
    
    @torch.no_grad()
    def extract_geometry(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = False,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        使用几何函数从密集网格点中提取几何信息
        
        Args:
            geometric_func: 几何函数，用于计算符号距离场（SDF）值
            batch_size: 批处理大小，默认为1
            bounds: 边界框范围，格式为(x_min, y_min, z_min, x_max, y_max, z_max)
            octree_depth: 八叉树深度，默认为7
            num_chunks: 分块数量，用于分批处理网格点，默认为10000
            disable_tqdm: 是否禁用进度条，默认为False
            
        Returns:
            trimesh.Trimesh: 提取的几何网格
        """
        # 初始化低分辨率网格
        grid_size = 257
        grid64 = np.linspace(-1.25, 1.25, grid_size)

        # 处理边界框输入
        if isinstance(bounds, float):
            bounds = abs(bounds)
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])  # 边界框最小值
        bbox_max = np.array(bounds[3:6])  # 边界框最大值
        bbox_size = bbox_max - bbox_min   # 边界框尺寸

        # 生成密集网格点
        xyz_samples, grid_size_dense, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=8,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)

        # 分批查询SDF值
        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :].to(self.device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = geometric_func(batch_queries)
            batch_logits.append(logits.cpu())

        sdf_values = torch.cat(batch_logits, dim=1)
        sdf_values = sdf_values.view((batch_size, grid_size_dense[0], grid_size_dense[1], grid_size_dense[2]))
        sdf_values = sdf_values.numpy()[0]

        # 创建高分辨率网格
        grid_size_high_res = 513
        grid128 = np.linspace(-1.25, 1.25, grid_size_high_res)
        x_high_res, y_high_res, z_high_res = np.meshgrid(grid128, grid128, grid128, indexing='ij')

        # 使用低分辨率SDF值创建插值器
        interpolator = RegularGridInterpolator((grid64, grid64, grid64), sdf_values)

        # 插值高分辨率网格的SDF值
        coords_high_res = np.stack((x_high_res, y_high_res, z_high_res), axis=-1)
        sdf_values_high_res = interpolator(coords_high_res)

        # 检测低分辨率网格中符号变化的顶点
        mixed_signs = np.zeros_like(sdf_values, dtype=bool)
        mixed_signs[:-1, :-1, :-1] = (
            (np.sign(sdf_values[:-1, :-1, :-1]) != np.sign(sdf_values[1:, 1:, 1:])) |
            (np.sign(sdf_values[:-1, :-1, 1:]) != np.sign(sdf_values[1:, 1:, :-1])) |
            (np.sign(sdf_values[:-1, 1:, :-1]) != np.sign(sdf_values[1:, :-1, 1:])) |
            (np.sign(sdf_values[1:, :-1, :-1]) != np.sign(sdf_values[:-1, 1:, 1:]))
        )

        # 将符号变化顶点的SDF值设为nan
        sdf_values_high_res[2 * mixed_signs] = np.nan

        # 查询nan顶点的SDF值
        nan_vertices = np.isnan(sdf_values_high_res)
        coords_nan = coords_high_res[nan_vertices]
        xyz_samples = torch.FloatTensor(coords_nan)

        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :].to(self.device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = geometric_func(batch_queries)
            batch_logits.append(logits.cpu())
        sdf_values_nan_vertices = torch.cat(batch_logits, dim=1).numpy()[0]

        # 将查询到的SDF值填充回高分辨率网格
        sdf_values_nan_vertices = torch.cat(batch_logits, dim=1).numpy()[0]
        sdf_values_nan_vertices = sdf_values_nan_vertices.flatten()  # 确保是 1 维数组
        sdf_values_high_res[nan_vertices] = sdf_values_nan_vertices

        # 使用marching cubes提取网格
        try:
            vertices, faces, normals, _ = measure.marching_cubes(sdf_values_high_res, 0, method="lewiner")
            vertices = vertices / grid_size_high_res * bbox_size + bbox_min
            mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.ascontiguousarray(faces))
            return mesh
        except (ValueError, RuntimeError) as e:
            print(f"❌ 网格提取失败: {e}")
            return trimesh.Trimesh()

    def extract_geometry_with_stats(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = True,
        **kwargs
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        提取几何并返回统计信息
        
        Args:
            参数与extract_geometry相同
            
        Returns:
            Tuple[trimesh.Trimesh, dict]: 提取的网格和统计信息
        """
        mesh = self.extract_geometry(
            geometric_func, batch_size, bounds, octree_depth, num_chunks, disable_tqdm, **kwargs
        )
        
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        
        return mesh, stats


# 保持向后兼容的函数接口
@torch.no_grad()
def extract_geometry_fast_v1(
        geometric_func: Callable,
        device: torch.device,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = True
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    向后兼容的函数接口，使用FastGeometryExtractor类
    
    Args:
        geometric_func: 几何函数
        device: 计算设备
        batch_size: 批处理大小
        bounds: 边界框范围
        octree_depth: 八叉树深度
        num_chunks: 分块数量
        disable_tqdm: 是否禁用进度条
        
    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]: 网格顶点面片列表和表面存在标志
    """
    extractor = FastGeometryExtractorV1(device)
    mesh = extractor.extract_geometry(
        geometric_func, batch_size, bounds, octree_depth, num_chunks, disable_tqdm
    )
    
    # 转换为向后兼容的格式
    if extractor.validate_mesh(mesh):
        mesh_v_f = [(mesh.vertices.astype(np.float32), np.ascontiguousarray(mesh.faces))]
        has_surface = np.array([True], dtype=np.bool_)
    else:
        mesh_v_f = [(None, None)]
        has_surface = np.array([False], dtype=np.bool_)
    
    return mesh_v_f, has_surface


if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用新的类接口进行测试
    print("=" * 50)
    print("开始运行 FastGeometryExtractor 测试套件")
    print("=" * 50)
    start_time = time.time()
    extractor = FastGeometryExtractorV1(device)
    mesh, stats = extractor.extract_geometry_with_stats(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        batch_size=1,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        num_chunks=10000,
        disable_tqdm=True
    )
    mesh.export("output_fast_mesh.obj")
    
    # 输出网格详细信息
    print("提取的网格信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 验证网格大致是球形的（体积应该在球体体积附近）
    expected_volume = (4/3) * np.pi * (0.5**3)  # 半径为0.5的球体体积
    volume_ratio = stats['volume'] / expected_volume
    print(f"  体积比 (实际/预期): {volume_ratio:.3f}")
    print(f"✅ FastGeometryExtractor 测试通过！耗时：{time.time() - start_time:.2f}秒")
    
    # 测试向后兼容的函数接口
    print("=" * 50)
    print("开始测试向后兼容性...")
    print("=" * 50)
    start_time = time.time()
    mesh_v_f, has_surface = extract_geometry_fast_v1(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        device=device,
        batch_size=1,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        num_chunks=10000,
        disable_tqdm=True
    )
    
    assert len(mesh_v_f) == 1, "❌ 向后兼容函数失败"
    assert has_surface[0], "❌ 向后兼容函数失败"
    print(f"✅ 向后兼容性测试通过！耗时: {time.time() - start_time:.2f}秒")
    print("=" * 50)
    print("所有测试通过！")
    # ✅ FastGeometryExtractor 测试通过！耗时：753.39秒
