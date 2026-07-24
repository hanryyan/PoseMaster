import trimesh
import os
import torch
import cv2

import numpy as np
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Base dir: {base_dir}")
sys.path.append(base_dir)

from structure import get_render_18, get_render_one
from PIL import Image
from ulip_score import ULIP
from uni3d_score import Uni3DScore

def NormalizeMesh(scene_mesh):
    bbox = scene_mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale = 1 / scale * 2 * 0.98

    scene_mesh.apply_translation(-center)
    scene_mesh.apply_scale(scale)

    # print(np.min(scene_mesh.vertices), np.max(scene_mesh.vertices))

    return scene_mesh, center, scale

def save_depth_map(depth, save_path, colormap=cv2.COLORMAP_JET, normalize=True):
    """
    使用OpenCV保存深度图
    
    参数:
    depth: numpy数组, shape (H, W, 1) 或 (H, W), 值范围 [0, 1]
    save_path: 保存路径
    colormap: OpenCV色彩映射，如 cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA 等
    normalize: 是否归一化到 [0, 1]
    """
    # 确保深度图是2D的
    if depth.ndim == 3:
        depth = depth.squeeze()
    
    # 如果需要归一化
    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # 转换为8位图像
    depth_uint8 = (depth * 255).astype(np.uint8)
    
    # 应用色彩映射
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 保存图像
    cv2.imwrite(save_path, depth_colored)
    # print(f"深度图已保存到: {save_path}")
    
    return depth_colored

def visualize_normal_map(normal_map, save_path=None):
    """
    可视化法线贴图
    
    参数:
    normal_map: numpy数组, shape (H, W, 3), 值范围通常为 [0, 1] 或 [-1, 1]
    save_path: 保存路径，如果为None则不保存
    """
    # 确保是numpy数组
    if not isinstance(normal_map, np.ndarray):
        normal_map = np.array(normal_map)
    
    # 检查法线贴图的数值范围并调整到[0, 1]
    if normal_map.min() < 0:
        # 假设范围是[-1, 1]，调整到[0, 1]
        normal_visual = (normal_map + 1) / 2
    else:
        # 假设范围已经是[0, 1]
        normal_visual = normal_map.copy()
    
    # 转换为8位图像
    normal_uint8 = (normal_visual * 255).astype(np.uint8)
    
    # 保存图像
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))
        # print(f"法线贴图已保存到: {save_path}")
    
    return normal_uint8

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def normal_map_cosine_similarity(normal_map1, normal_map2, mask=None):
    """
    计算两个法线贴图的余弦相似度
    
    参数:
    normal_map1: 第一个法线贴图，形状为(H, W, 3)的numpy数组，值范围通常为[0,1]或[0,255]
    normal_map2: 第二个法线贴图，形状与normal_map1相同
    mask: 可选掩码，形状为(H, W)，指定要计算相似度的区域
    
    返回:
    similarity: 平均余弦相似度
    similarity_map: 逐像素的余弦相似度图
    """
    
    # 确保输入是numpy数组
    normal_map1 = np.array(normal_map1, dtype=np.float32)
    normal_map2 = np.array(normal_map2, dtype=np.float32)
    
    # 将法线贴图归一化到[-1, 1]范围
    if normal_map1.max() > 1.0:
        normal_map1 = normal_map1 / 255.0 * 2.0 - 1.0
    else:
        normal_map1 = normal_map1 * 2.0 - 1.0
        
    if normal_map2.max() > 1.0:
        normal_map2 = normal_map2 / 255.0 * 2.0 - 1.0
    else:
        normal_map2 = normal_map2 * 2.0 - 1.0
    
    # 验证形状
    if normal_map1.shape != normal_map2.shape:
        raise ValueError(f"法线贴图形状不匹配: {normal_map1.shape} vs {normal_map2.shape}")
    
    # 归一化法线向量（确保单位长度）
    norm1 = np.linalg.norm(normal_map1, axis=2, keepdims=True)
    norm2 = np.linalg.norm(normal_map2, axis=2, keepdims=True)
    
    # 避免除以零
    norm1 = np.where(norm1 == 0, 1e-8, norm1)
    norm2 = np.where(norm2 == 0, 1e-8, norm2)
    
    normal_map1_normalized = normal_map1 / norm1
    normal_map2_normalized = normal_map2 / norm2
    
    # 计算逐像素余弦相似度
    similarity_map = np.sum(normal_map1_normalized * normal_map2_normalized, axis=2)
    
    # 应用掩码（如果有）
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        valid_pixels = similarity_map[mask]
    else:
        valid_pixels = similarity_map.flatten()
    
    # 计算平均相似度
    mean_similarity = np.mean(valid_pixels)
    
    return mean_similarity, similarity_map

def mean_angular_error(pred_norm, gt_norm, mask=None):
    if len(pred_norm.shape) == 4:
        pred_norm = pred_norm.squeeze(0)
    if len(gt_norm.shape) == 4:
        gt_norm = gt_norm.squeeze(0)

    # shape must be [3,H,W]
    assert (gt_norm.shape[0] == 3) and (
        pred_norm.shape[0] == 3
    ), "Channel dim should be the first dimension!"
    # mask out the zero vectors, otherwise torch.cosine_similarity computes 90° as error
    if mask is not None:
        ch, h, w = gt_norm.shape

        mask = torch.norm(gt_norm, dim=0) > 0

        pred_norm = pred_norm[:, mask.view(h, w)]
        gt_norm = gt_norm[:, mask.view(h, w)]

    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=0)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi  # (H, W)
    cosine_error = pred_error.view(-1).detach().cpu().numpy()

    return round(np.average(cosine_error), 4)

pre_meshes = []
gt_meshes = []


ulip = ULIP()
uni3d = Uni3DScore()

task_dir = "cvpr_experiments/vs_apose"
methods = ["ours"]
for method in methods:
    # print(f"Evaluating method: {method}")
    metrics_dict = {
        "angular_error": [],
        "normal_similarity": [],
        "uni3d": [],
        "ulip": []
    }
    save_dir = os.path.join(task_dir, method)
    import glob
    gt_files = glob.glob(os.path.join(task_dir, "gt_mesh", "*.obj"))
    for file_name in gt_files:
        uid = file_name.split("/")[-1].split(".")[0]
        # print(f"uid: {uid}")
        gt_mesh_file = os.path.join(task_dir, "gt_mesh", file_name)
        gt_image = os.path.join(task_dir, "gt_img", f"{uid}.png")
        pre_mesh_file = os.path.join(task_dir, method, f"{uid}.glb")
        # print(f"pre_mesh_file: {pre_mesh_file}")
        gt_mesh = trimesh.load(gt_mesh_file, force='mesh')
        pre_mesh = trimesh.load(pre_mesh_file, force='mesh')
        if "ply" in gt_mesh_file:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                                    angle=np.radians(-90),
                                    direction=[1, 0, 0]
                                )
            gt_mesh.apply_transform(rotation_matrix)
        if method in ["ChracterGen"]:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                                    angle=np.radians(-180),
                                    direction=[0, 1, 0]
                                )
            pre_mesh.apply_transform(rotation_matrix)
        
        pre_mesh, _, _ = NormalizeMesh(pre_mesh)
        gt_mesh, _, _ = NormalizeMesh(gt_mesh)
        gt_normals, gt_depths, gt_masks = get_render_18(gt_mesh.vertices,  gt_mesh.faces, "cuda", img_size=512)
        # print(f"gt_normals shape: {gt_normals.shape}")
        pred_normal, pre_depth, pre_mask = get_render_one(pre_mesh.vertices,  pre_mesh.faces, "cuda", img_size=512)
        # save depth map
        best_normal_similarity = 0
        best_angular_error = float('inf')
        best_normal = None
        for j in range(18):
            # print(f"pred_normal depth: {pred_normal.permute(2, 0, 1).contiguous().shape}, gt_normals {gt_normals[j].permute(2, 0, 1).contiguous().shape}")
            angular_error = mean_angular_error(pred_normal.permute(2, 0, 1).contiguous(), gt_normals[j].permute(2, 0, 1).contiguous())
            normal_similarity, similarity_map = normal_map_cosine_similarity(pred_normal.cpu().numpy(), gt_normals[j].cpu().numpy())
            # visualize_normal_map(gt_normals[j, ...].cpu().numpy(), os.path.join(task_dir, "gt_mesh_normal", uid+"_"+str(j)+"_normal.png"))
            # print(f"uid:{uid}, idx: {j}, normal_similarity: {normal_similarity}")
            if normal_similarity > best_normal_similarity:
                best_normal_similarity = normal_similarity
                best_normal = gt_normals[j, ...]
            if angular_error < best_angular_error:
                best_angular_error = angular_error
                

        metrics_dict["normal_similarity"].append(best_normal_similarity)
        metrics_dict["angular_error"].append(best_angular_error)
        # visualize_normal_map(pred_normal.cpu().numpy(), os.path.join(task_dir, method, uid+"_normal.png"))

        # visualize_normal_map(best_normal.cpu().numpy(), os.path.join(task_dir, "gt_mesh_normal", uid+"_"+method+"_normal.png"))

        samples, _ = trimesh.sample.sample_surface(pre_mesh, 8192)
        pc = torch.from_numpy(samples).float().unsqueeze(0)
        images = [Image.open(gt_image).convert('RGB')]
        uni3d_score = uni3d.sim_img(pc, images).item()
        
        ulip_score = ulip.sim_img(pc, images).item()
        # print(f"File: {file_name}, normal_similarity: {normal_similarity}, uni3d: {uni3d_score}, ulip: {ulip_score}")
        metrics_dict["uni3d"].append(uni3d_score)
        metrics_dict["ulip"].append(ulip_score)

    metrics_dict["normal_similarity"] = sum(metrics_dict["normal_similarity"]) / len(metrics_dict["normal_similarity"])
    metrics_dict["angular_error"] = sum(metrics_dict["angular_error"]) / len(metrics_dict["angular_error"])
    metrics_dict["uni3d"] = sum(metrics_dict["uni3d"]) / len(metrics_dict["uni3d"])
    metrics_dict["ulip"] = sum(metrics_dict["ulip"]) / len(metrics_dict["ulip"])
    print(f"Method: {method}, angular_error: {metrics_dict['angular_error']} normal_similarity: {metrics_dict['normal_similarity']}, uni3d: {metrics_dict['uni3d']}, ulip: {metrics_dict['ulip']}")
    result = f'angular_error: {metrics_dict["angular_error"]}\n normal_similarity: {metrics_dict["normal_similarity"]}\n uni3d: {metrics_dict["uni3d"]}\n ulip: {metrics_dict["ulip"]}'
    with open(os.path.join(save_dir, "results.txt"), "w") as f:
        f.write(result)

