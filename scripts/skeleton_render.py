import numpy as np
from vedo import Plotter, Sphere, Cone, settings
import os

bone_order_names_playbox = ['目.L', '目.R', '頭', '首', '肩.L', '腕.L', 
                              'ひじ.L', '手首.L', '親指０.L', '親指１.L', '親指２.L', '人指１.L', '人指２.L', '人指３.L',
                              '中指１.L', '中指２.L', '中指３.L', '薬指１.L', '薬指２.L', '薬指３.L', '小指１.L', '小指２.L',
                              '小指３.L', '肩.R', '腕.R', 'ひじ.R', '手首.R', '親指０.R', '親指１.R', '親指２.R', '人指１.R', '人指２.R', '人指３.R',
                               '中指１.R', '中指２.R', '中指３.R', '薬指１.R', '薬指２.R', '薬指３.R', '小指１.R', '小指２.R',
                               '小指３.R', '上半身2', '上半身', '下半身', '足.L', 'ひざ.L', 
                              '足首.L', '足.R', 'ひざ.R', '足首.R'
                              ]


bone_names_vroid = ['J_Bip_C_Hips', 'J_Bip_C_Spine', 'J_Bip_C_Chest', 'J_Bip_C_UpperChest', 'J_Bip_C_Neck', 'J_Bip_C_Head', 'J_Adj_L_FaceEye', 
                               'J_Adj_R_FaceEye', 'J_Bip_L_Shoulder', 'J_Bip_L_UpperArm', 'J_Bip_L_LowerArm', 'J_Bip_L_Hand', 'J_Bip_R_Shoulder', 'J_Bip_R_UpperArm',
                               "J_Bip_R_LowerArm", "J_Bip_R_Hand", "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg", "J_Bip_L_Foot", 'J_Bip_R_UpperLeg',
                               "J_Bip_R_LowerLeg", "J_Bip_R_Foot"]      

bone_names = bone_order_names_playbox

def get_normalization_parameters(points, method='sphere'):
    """
    获取点云归一化参数
    
    参数:
    points: numpy数组, 形状为(N, 3)
    method: 归一化方法
        'sphere': 归一化到单位球体（默认）
        'cube': 归一化到单位立方体
        'standard': 标准化（零均值，单位方差）
    
    返回:
    normalized_points: 归一化后的点云
    scale: 缩放参数
    offset: 偏移参数
    """
    points = np.array(points)
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("输入点云形状应为(N, 3)")
    
    if method == 'sphere':
        # 方法1：归一化到单位球体
        offset = np.mean(points, axis=0)  # 质心作为偏移
        centered = points - offset
        scale = np.max(np.linalg.norm(centered, axis=1))  # 最大距离作为缩放
        if scale == 0:
            scale = 1.0
        normalized_points = centered / scale
        
    elif method == 'cube':
        # 方法2：归一化到单位立方体 [-0.5, 0.5]^3
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        offset = min_vals  # 最小值作为偏移
        scale = max_vals - min_vals  # 范围作为缩放
        scale[scale == 0] = 1.0  # 避免除以零
        normalized_points = (points - offset) / scale - 0.5
        
    elif method == 'standard':
        # 方法3：标准化（零均值，单位方差）
        offset = np.mean(points, axis=0)  # 均值作为偏移
        scale = np.std(points, axis=0)  # 标准差作为缩放
        scale[scale == 0] = 1.0  # 避免除以零
        normalized_points = (points - offset) / scale
        
    else:
        raise ValueError("方法必须是 'sphere', 'cube' 或 'standard'")
    
    return normalized_points, scale, offset

def load_skeleton(skeleton_data_path, num_joints=51, is_transform=True):
    data = np.load(skeleton_data_path, allow_pickle=True)
    body = data['body']
    lefthand = data["lefthand"]
    righthand = data["righthand"]
    if len(body) < 16:
        raise ValueError("no human skeleton") 
    bone_start = []
    bone_end = []
    bone_name_dict = {}
    for idx, (keypoint, name) in enumerate(body + lefthand + righthand):
        # print(name)
        if name in bone_names and name not in bone_name_dict.keys():
            bone_name_dict[name] = 0
            bone_start.append(keypoint[0])
            bone_end.append(keypoint[1])

    bone_start = np.array(bone_start).astype(np.float32)
    bone_end = np.array(bone_end).astype(np.float32)
    
    bone_start, scale, offset = get_normalization_parameters(bone_start, method='sphere')
    bone_end = (bone_end - offset) / scale


    if bone_start.shape[0] < num_joints:
        # padding zero
        zeros = np.zeros((num_joints - bone_start.shape[0], 3))
        bone_start = np.concatenate((bone_start, zeros), axis=0)
        bone_end = np.concatenate((bone_end, zeros), axis=0)

    ret = {'bone_start': bone_start, "bone_end": bone_end}
    return ret

# 设置中文字体
settings.default_font = 'Arial'  # 确保支持中文显示的字体

def create_joint(position, radius=0.1, color='white'):
    """创建关节点(球体)"""
    return Sphere(pos=position, r=radius, c=color)

def create_bone(start_position, end_position, radius=0.05, color='white'):
    """创建骨骼连接(圆锥)，顶点位于起点，底面位于终点"""
    # 计算方向向量
    direction = np.array(end_position) - np.array(start_position)
    length = np.linalg.norm(direction)
    
    if length < 1e-10:  # 避免长度接近零的情况
        return None
    
    # 圆锥的位置应该是底面的中心，而不是顶点
    # 所以我们需要从终点往回移动半个圆锥高度
    base_center = np.array(end_position) - (direction / 2)
    
    # 创建圆锥（顶点位于起点，底面中心位于base_center）
    bone = Cone(pos=base_center, 
                r=radius, 
                height=length, 
                axis=direction, 
                c=color)
    
    return bone

def visualize_skeleton(bone_start_points, bone_end_points, output_dir='skeleton_views', 
                       joint_radius=0.02, bone_radius=0.02):
    """可视化骨骼系统并从6个不同视角保存图像
    
    参数:
    bone_start_points: 21个骨骼的起点坐标列表，每个元素是一个三元组(x, y, z)
    bone_end_points: 21个骨骼的终点坐标列表，每个元素是一个三元组(x, y, z)
    output_dir: 保存图像的目录
    joint_radius: 关节点(球体)的半径
    bone_radius: 骨骼(圆锥)的底部半径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义6个不同的视角
    # view_angles = [
    #     dict(azimuth=0, elevation=0, roll=0, description="前视图"),
    #     dict(azimuth=180, elevation=0, roll=0, description="后视图"),
    #     dict(azimuth=90, elevation=0, roll=0, description="右视图"),
    #     dict(azimuth=270, elevation=0, roll=0, description="左视图"),
    #     dict(azimuth=0, elevation=90, roll=0, description="顶视图"),
    #     dict(azimuth=0, elevation=-90, roll=0, description="底视图"),
    # ]

    view_angles = [
        dict(azimuth=i*30, elevation=0, roll=0) for i in range(12)
    ]
    
    # 从每个视角渲染并保存图像
    for i, view in enumerate(view_angles):
        # 创建一个Plotter对象，启用离线渲染
        plt = Plotter(offscreen=True, axes=0, interactive=False)
        plt.renderer.SetBackground(1, 1, 1)
        
        lens = len(bone_start_points)
        for idx in range(lens):
            bone_start_point = bone_start_points[idx]
            bone_end_point = bone_end_points[idx]
            start_joint = create_joint(bone_start_point, radius=joint_radius)
            plt += start_joint
            end_joint = create_joint(bone_end_point, radius=joint_radius)
            plt += end_joint
        
            bone = create_bone(bone_start_point, bone_end_point, radius=bone_radius)
            if bone:
                plt += bone
        
        plt.reset_camera()
        # 自动调整相机位置以包含所有对象
        plt.show(azimuth=view['azimuth'], elevation=view['elevation'], 
                 roll=view['roll'], interactive=False)
        
        # 保存带有透明背景的图像
        plt.export(filename='scene.obj', binary=False)

        filename = os.path.join(output_dir, f"skeleton_view_{i+1}.png")
        plt.screenshot(filename)
        # print(f"已保存视角 {i+1}: {view['description']} 到 {filename}")
        
        plt.close()
    

if __name__ == "__main__":
    skeletons = [
        "val_data/Apose/45_bone_points.pkl"
    ]
    save_dir = "cvpr_experiments/vs_apose/skeleton"
    for idx, skeleton in enumerate(skeletons):
        ret = load_skeleton(skeleton, is_transform=False)
        bone_start_points = ret['bone_start']
        bone_end_points = ret['bone_end']
        visualize_skeleton(bone_start_points, bone_end_points, output_dir=os.path.join(save_dir, str(idx)))