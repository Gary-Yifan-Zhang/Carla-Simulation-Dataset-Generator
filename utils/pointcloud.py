"""
整合后的点云处理程序
功能：
1. 多雷达点云融合
2. 点云可视化
3. 边界框处理
4. 批量处理
"""

import os
import yaml
import numpy as np
import open3d as o3d

def load_config(config_path):
    """从yaml文件加载雷达配置"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    lidar_order = {
        'LIDAR': 0,        # 主雷达
        'SUB_LIDAR_1': 1,  # 子雷达1
        'SUB_LIDAR_2': 2,  # 子雷达2
        'SUB_LIDAR_3': 3,  # 子雷达3
        'SUB_LIDAR_4': 4   # 子雷达4
    }

    lidar_configs = {}
    for key in config['SENSOR_CONFIG']:
        if key in lidar_order:
            lidar_configs[key] = {
                'transform': config['SENSOR_CONFIG'][key]['TRANSFORM'],
                'index': lidar_order[key]
            }
    return lidar_configs

def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角转换为旋转矩阵"""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
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
    """应用坐标变换到点云"""
    location = np.array(transform_config['location'])
    location[1] *= -1
    location[2] -= 1.6
    
    rotation = np.array(transform_config['rotation'])
    rotation[0] *= -1

    R = euler_to_rotation_matrix(*rotation)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = location
    
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (T @ homogeneous_points.T).T[:, :3]
    
    return transformed_points

def read_point_cloud(file_path):
    """读取二进制点云文件"""
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    # print(f"读取点云：{file_path}，形状：{point_cloud.shape}")
    return point_cloud

def read_calibration(file_path):
    """读取标定信息"""
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros(3)

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam:'):
                values = list(map(float, line.split()[1:]))
                if len(values) == 12:
                    rotation_matrix = np.array([
                        values[0], values[1], values[2],
                        values[4], values[5], values[6],
                        values[8], values[9], values[10]
                    ]).reshape(3, 3)
                    translation_vector = np.array([values[3], values[7], values[11]])
    return rotation_matrix, translation_vector

def read_bounding_boxes(file_path, calibration_matrix, translation_vector):
    """读取并处理边界框"""
    bboxes = []
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            object_type = data[0]
            h, l, w = float(data[9]), float(data[10]), float(data[11])
            
            # # 尺寸调整
            # h *= 1.05
            # w *= 1.05
            # l *= 1.05
            
            x, y, z = float(data[12]), float(data[13]), float(data[14])
            rotation_y = float(data[15])

            bbox = create_bbox(x, y, z, h, w, l, rotation_y, object_type, calibration_matrix, translation_vector)
            
            metadata[bbox] = {
                'object_type': object_type,
                'original_center': np.array([x, y, z]),
                'original_extent': [h, w, l]
            }
            bboxes.append(bbox)
    return bboxes, metadata

def create_bbox(x, y, z, h, w, l, rotation_y, object_type, calibration_matrix, translation_vector):
    """创建边界框"""
    TYPE_COLORS = {
        "Pedestrian": (1, 0, 0),
        "Car": (0, 1, 0),
        "Bicycle": (1, 1, 0),
        "TrafficLight": (0, 0, 1),
        "TrafficSigns": (0, 0, 1)
    }

    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.extent = [l, w, h]
    
    z_offset = h / 2 if object_type in ["Pedestrian", "Car", "Bicycle"] else 0
    bbox.center = np.array([x, y, z + z_offset])
    bbox.color = TYPE_COLORS.get(object_type, (0, 0, 1))

    theta = rotation_y
    R_obj = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    bbox.rotate(R_obj, center=bbox.center)

    return bbox

def merge_point_clouds(data_folder, file_id, lidar_configs):
    """融合多雷达点云"""
    merged_points = []
    
    for lidar_name, config in lidar_configs.items():
        file_path = f"{data_folder}/velodyne/{file_id}_lidar_{config['index']}.bin"
        points = read_point_cloud(file_path)
        
        if config['index'] == 0:
            transformed_points = points
        else:
            transformed_points = transform_point_cloud(points[:, :3], config['transform'])
            transformed_points = np.hstack([transformed_points, points[:, 3:]])
        
        merged_points.append(transformed_points)
    
    return np.vstack(merged_points)

def visualize(point_cloud, bboxes):
    """可视化点云和边界框"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis] + bboxes)

def batch_merge(data_folder, config_path):
    """批量处理所有帧"""
    lidar_configs = load_config(config_path)
    velodyne_dir = os.path.join(data_folder, "velodyne")
    file_ids = set(f.split("_")[0] for f in os.listdir(velodyne_dir) if f.endswith(".bin"))
    
    for file_id in sorted(file_ids):
        print(f"\n正在处理帧：{file_id}")
        
        calibration_file_path = f"{data_folder}/calib/{file_id}.txt"
        rotation_matrix, translation_vector = read_calibration(calibration_file_path)
        
        bboxes, _ = read_bounding_boxes(f"{data_folder}/lidar_label/{file_id}.txt", 
                                      rotation_matrix, translation_vector)
        
        merged_pc = merge_point_clouds(data_folder, file_id, lidar_configs)
        
        output_path = f"{data_folder}/velodyne/{file_id}_lidar_999.bin"
        merged_pc.astype(np.float32).tofile(output_path)
        print(f"融合点云已保存至：{output_path}")

if __name__ == "__main__":
    config_path = "configs.yaml"
    data_folder = "data/training_20250402_173234"
    file_id = "000001"
    
    lidar_configs = load_config(config_path)
    calibration_file_path = f"{data_folder}/calib/{file_id}.txt"
    rotation_matrix, translation_vector = read_calibration(calibration_file_path)
    
    bboxes, _ = read_bounding_boxes(f"{data_folder}/lidar_label/{file_id}.txt", 
                                  rotation_matrix, translation_vector)
    
    merged_pc = merge_point_clouds(data_folder, file_id, lidar_configs)
    visualize(merged_pc, bboxes)
    
    batch_merge(data_folder, config_path)