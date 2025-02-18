import numpy as np
from view_pc import read_point_cloud
import re

# 雷达位姿信息
LIDAR_POSES = {
    'LIDAR': {
        'location': np.array([0, 0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    },
    'SUB_LIDAR_1': {
        'location': np.array([0.0, 0.8, 1.6]), # 0.8
        'rotation': np.eye(3)  
    },
    'SUB_LIDAR_2': {
        'location': np.array([0.0, -0.8, 1.6]), # -0.8
        'rotation': np.eye(3) 
    },
    'SUB_LIDAR_3': {
        'location': np.array([-1.0, 0.8, 1.6]), # 0.8
        'rotation': np.eye(3)  
    },
    'SUB_LIDAR_4': {
        'location': np.array([-1.0, -0.8, 1.6]), # -0.8
        'rotation': np.eye(3) 
    }
}

# # 雷达位姿信息
# LIDAR_POSES = {
#     'LIDAR': {
#         'location': np.array([0, 0, 1.6]),
#         'rotation': np.eye(3)  # 无旋转
#     },
#     'SUB_LIDAR_1': {
#         'location': np.array([-0.8, 0.0, 1.6]),
#         'rotation': np.eye(3)  # 无旋转
#     },
#     'SUB_LIDAR_2': {
#         'location': np.array([0.8, 0.0, 1.6]),
#         'rotation': np.eye(3)  # 无旋转
#     },
#     'SUB_LIDAR_3': {
#         'location': np.array([-0.8, -1.0, 1.6]),
#         'rotation': np.eye(3)  # 无旋转
#     },
#     'SUB_LIDAR_4': {
#         'location': np.array([0.8, -1.0, 1.6]),
#         'rotation': np.eye(3)  # 无旋转
#     }
# }

def transform_point_cloud(points, rotation_matrix, translation_vector):
    """
    将点云转换到目标坐标系
    :param points: 原始点云 (N, 3)
    :param rotation_matrix: 旋转矩阵 (3, 3)
    :param translation_vector: 平移向量 (3,)
    :return: 转换后的点云 (N, 3)
    """
    return np.dot(points, rotation_matrix.T) + translation_vector

def euler_to_rotation_matrix(pitch, yaw, roll):
    """
    将欧拉角转换为旋转矩阵（ZYX顺序，单位：弧度）
    """
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    R = np.array([
        [cy*cr + sy*sp*sr, -cy*sr + sy*sp*cr, sy*cp],
        [cp*sr, cp*cr, -sp],
        [-sy*cr + cy*sp*sr, sy*sr + cy*sp*cr, cy*cp]
    ])
    return R

def read_ego_motion(file_path):
    """
    读取ego_state文件获取车辆位姿信息
    :return: (rotation_matrix, translation_vector)
    """
    with open(file_path, 'r') as f:
        content = f.read()
        
    # 解析位置和旋转
    loc_match = re.search(r'Location\(x=([\d.-]+),\s*y=([\d.-]+),\s*z=([\d.-]+)\)', content)
    rot_match = re.search(r'Rotation\(pitch=([\d.-]+),\s*yaw=([\d.-]+),\s*roll=([\d.-]+)\)', content)
    
    translation = np.array([float(loc_match.group(1)), 
                           float(loc_match.group(2)), 
                           float(loc_match.group(3))])
    
    # 将角度转换为弧度
    pitch = np.deg2rad(float(rot_match.group(1)))
    yaw = np.deg2rad(float(rot_match.group(2)))
    roll = np.deg2rad(float(rot_match.group(3)))
    
    rotation = euler_to_rotation_matrix(pitch, yaw, roll)
    return rotation, translation

def merge_point_clouds(data_folder, file_id):
    """
    合并指定ID的所有点云到全局坐标系
    :param data_folder: 数据文件夹路径
    :param file_id: 文件ID
    :return: 合并后的点云
    """
    
    # 读取车辆全局位姿
    ego_file = f"{data_folder}/ego_state/{file_id}.txt"
    vehicle_rotation, vehicle_translation = read_ego_motion(ego_file)
    
    # 读取主雷达的点云
    merged_pc = read_point_cloud(f"{data_folder}/velodyne/{file_id}_lidar_0.bin")
    
    # 合并所有雷达的点云
    for i, lidar_key in enumerate(LIDAR_POSES.keys(), start=0):
        if lidar_key == 'LIDAR' and i != 0:
            continue
            
        # 读取当前雷达的点云
        curr_pc = read_point_cloud(f"{data_folder}/velodyne/{file_id}_lidar_{i}.bin")
        
        # 获取雷达相对于车辆的变换
        lidar_rot = LIDAR_POSES[lidar_key]['rotation']
        lidar_trans = LIDAR_POSES[lidar_key]['location']
        lidar_trans[2] = 0
        
        # 组合变换：车辆全局位姿 * 雷达相对位姿
        combined_rotation = np.dot(vehicle_rotation, lidar_rot)
        # combined_translation = np.dot(vehicle_rotation, lidar_trans) + vehicle_translation
        combined_translation = np.dot(vehicle_rotation, lidar_trans) 
        
        # 转换点云到全局坐标系
        transformed_pc = transform_point_cloud(curr_pc, combined_rotation, combined_translation)
        
        # 合并点云
        merged_pc = np.vstack((merged_pc, transformed_pc))
    
    return merged_pc

def visualize_point_cloud(points):
    """
    可视化点云
    :param points: 点云数据 (N, 3)
    """
    # 这里使用open3d进行点云可视化
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def save_point_cloud(points, file_path):
    """
    保存点云数据到文件
    :param points: 点云数据 (N, 3)
    :param file_path: 保存路径
    """
    # 添加强度值（默认为0），将点云扩展为 (N, 4)
    points_with_intensity = np.hstack((points, np.zeros((points.shape[0], 1))))
    
    # 将点云展平为一维数组并保存
    flattened_points = points_with_intensity.astype(np.float32).flatten()
    flattened_points.tofile(file_path)
    print(f"Point cloud shape: ({flattened_points.shape[0]},)")


if __name__ == "__main__":
    data_folder = "data/training"
    file_id = "000009"
    
    # 合并点云
    merged_pc = merge_point_clouds(data_folder, file_id)
    
    # 可视化合并后的点云
    visualize_point_cloud(merged_pc)
    
    # 保存合并后的点云为index999
    output_path = f"{data_folder}/velodyne/{file_id}_lidar_999.bin"
    save_point_cloud(merged_pc, output_path)
    print(f"合并后的点云已保存至: {output_path}")
