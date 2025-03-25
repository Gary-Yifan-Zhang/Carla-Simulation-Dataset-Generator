"""
多雷达点云融合程序
功能：
1. 从config读取所有雷达外参
2. 将子雷达点云转换到主雷达坐标系
3. 融合显示所有点云
"""

import yaml
import numpy as np
from view_pc import read_point_cloud, visualize

def load_config(config_path):
    """从yaml文件加载雷达配置（适配新文件名格式）"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 定义雷达顺序映射
    lidar_order = {
        'LIDAR': 0,        # 主雷达
        'SUB_LIDAR_1': 1,  # 子雷达1
        'SUB_LIDAR_2': 2,   # 子雷达2
        'SUB_LIDAR_3': 3,  # 子雷达1
        'SUB_LIDAR_4': 4   # 子雷达2
    }

    lidar_configs = {}
    for key in config['SENSOR_CONFIG']:
        if key in lidar_order:
            lidar_configs[key] = {
                'transform': config['SENSOR_CONFIG'][key]['TRANSFORM'],
                'index': lidar_order[key]  # 新增索引字段
            }
    return lidar_configs

def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角转换为旋转矩阵"""
    # 转换为弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # 计算各轴旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def transform_point_cloud(points, transform_config):
    """应用坐标变换到点云（增加Y轴取反）"""
    # 解析变换参数并镜像Y轴
    location = np.array(transform_config['location'])
    location[1] *= -1  # Y轴位置取反
    location[2] -= 1.6  # 新增Z轴下移
    
    rotation = np.array(transform_config['rotation'])
    rotation[0] *= -1  # 滚转角取反（根据右手定则调整）

    # 构建变换矩阵（增加Y轴镜像）
    R = euler_to_rotation_matrix(*rotation)
    # # 创建Y轴镜像矩阵
    # mirror_y = np.array([
    #     [1,  0, 0],
    #     [0, -1, 0],  # Y轴取反
    #     [0,  0, 1]
    # ])
    # R = R @ mirror_y  # 组合旋转和镜像
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = location
    
    # 齐次坐标转换
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (T @ homogeneous_points.T).T[:, :3]
    
    return transformed_points

def merge_point_clouds(data_folder, file_id, lidar_configs):
    """融合多雷达点云（适配新文件名格式）"""
    merged_points = []
    
    for lidar_name, config in lidar_configs.items():
        # 生成新格式文件路径
        file_path = f"{data_folder}/velodyne/{file_id}_lidar_{config['index']}.bin"
        
        # 读取原始点云（添加调试信息）
        print(f"正在读取雷达 {config['index']}: {file_path}")
        points = read_point_cloud(file_path)
        
        # 应用坐标变换（主雷达不需要变换）
        if config['index'] == 0:
            transformed_points = points  # 主雷达保持原坐标
        else:
            transformed_points = transform_point_cloud(points, config['transform'])
        
        merged_points.append(transformed_points)
    
    return np.vstack(merged_points)

if __name__ == "__main__":
    # 配置参数
    config_path = "configs.yaml"
    data_folder = "data/training_20250325_142657"
    file_id = "000010"
    
    # 加载配置
    lidar_configs = load_config(config_path)
    
    # 融合点云
    merged_pc = merge_point_clouds(data_folder, file_id, lidar_configs)
    
    # 可视化（使用view_pc中的函数）
    visualize(merged_pc, [])