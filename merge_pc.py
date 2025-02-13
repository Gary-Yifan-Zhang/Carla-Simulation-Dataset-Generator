import numpy as np
from view_pc import read_point_cloud

# 雷达位姿信息
LIDAR_POSES = {
    'LIDAR': {
        'location': np.array([0, 0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    },
    'SUB_LIDAR_1': {
        'location': np.array([-0.8, 0.0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    },
    'SUB_LIDAR_2': {
        'location': np.array([0.8, 0.0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    },
    'SUB_LIDAR_3': {
        'location': np.array([-0.8, -1.0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    },
    'SUB_LIDAR_4': {
        'location': np.array([0.8, -1.0, 1.6]),
        'rotation': np.eye(3)  # 无旋转
    }
}
def transform_point_cloud(points, rotation_matrix, translation_vector):
    """
    将点云转换到目标坐标系
    :param points: 原始点云 (N, 3)
    :param rotation_matrix: 旋转矩阵 (3, 3)
    :param translation_vector: 平移向量 (3,)
    :return: 转换后的点云 (N, 3)
    """
    return np.dot(points, rotation_matrix.T) + translation_vector

def merge_point_clouds(data_folder, file_id):
    """
    合并指定ID的所有点云到主雷达坐标系
    :param data_folder: 数据文件夹路径
    :param file_id: 文件ID
    :return: 合并后的点云
    """
    # 读取主雷达的点云
    merged_pc = read_point_cloud(f"{data_folder}/velodyne/{file_id}_lidar_0.bin")
    print(f"主雷达点云形状: {merged_pc.shape}")
    
    # 合并其他雷达的点云
    for i, lidar_key in enumerate(['SUB_LIDAR_1', 'SUB_LIDAR_2', 'SUB_LIDAR_3', 'SUB_LIDAR_4'], start=1):
        # 读取当前雷达的点云
        curr_pc = read_point_cloud(f"{data_folder}/velodyne/{file_id}_lidar_{i}.bin")
        
        # 获取当前雷达相对于主雷达的位姿
        relative_translation = LIDAR_POSES[lidar_key]['location'] - LIDAR_POSES['LIDAR']['location']
        relative_rotation = np.dot(LIDAR_POSES['LIDAR']['rotation'], LIDAR_POSES[lidar_key]['rotation'].T)
        
        # 转换点云到主雷达坐标系
        transformed_pc = transform_point_cloud(curr_pc, relative_rotation, relative_translation)
        
        # 合并点云
        merged_pc = np.vstack((merged_pc, transformed_pc))
        
    print(f"合并后的点云形状: {merged_pc.shape}")
    
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
    file_id = "000026"
    
    # 合并点云
    merged_pc = merge_point_clouds(data_folder, file_id)
    
    # 可视化合并后的点云
    visualize_point_cloud(merged_pc)
    
    # 保存合并后的点云为index999
    output_path = f"{data_folder}/velodyne/{file_id}_lidar_999.bin"
    save_point_cloud(merged_pc, output_path)
    print(f"合并后的点云已保存至: {output_path}")
