"""
该程序用于读取和可视化点云数据及其对应的边界框。

功能：
- 从指定的二进制文件中读取点云数据。
- 从标定文件中读取激光雷达到相机的旋转矩阵和平移向量。
- 从边界框文件中读取物体的边界框信息。
- 将点云数据和边界框可视化。

使用方法：
1. 修改 `data_folder` 和 `file_id` 变量以指定数据文件夹和文件ID。
2. 运行程序以可视化点云和边界框。

注意：确保数据文件的路径和格式正确。
"""

import open3d as o3d
import numpy as np

def read_point_cloud(file_path):
    """
    读取二进制点云文件并返回点云数据。

    参数：
        file_path: 点云文件的路径。

    返回：
        point_cloud: 读取的点云数据，形状为 (N, 3)，只包含 x, y, z 坐标。
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # 只取前三列（x, y, z）

def read_bounding_boxes(file_path, calibration_matrix, translation_vector):
    """
    读取边界框文件并应用平移和旋转变换。

    参数：
        file_path: 边界框文件的路径。
        calibration_matrix: 从激光雷达到相机的旋转矩阵。
        translation_vector: 从激光雷达到相机的平移向量。

    返回：
        bboxes: 应用变换后的边界框列表。
    """
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            # 假设数据格式为：type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y
            object_type = data[0]  # 获取对象类型
            h, w, l = float(data[8]), float(data[9]), float(data[10])
            x, y, z = float(data[11]), float(data[12]), float(data[13])
            rotation_y = float(data[14])

            # 创建边界框
            bbox = create_bbox(x, y, z, h, w, l, rotation_y, object_type, calibration_matrix, translation_vector)

            bboxes.append(bbox)
    return bboxes

def read_calibration(file_path):
    """
    从文件中读取标定信息，包括旋转矩阵和平移向量。

    参数：
        file_path: 标定文件的路径。

    返回：
        rotation_matrix: 从激光雷达到相机的旋转矩阵。
        translation_vector: 从激光雷达到相机的平移向量。
    """
    rotation_matrix = np.eye(3)  # 初始化为3x3单位矩阵
    translation_vector = np.zeros(3)  # 初始化为3x1零向量

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam:'):
                values = list(map(float, line.split()[1:]))  # 读取标定矩阵的值
                if len(values) == 12:
                    # 使用指定的索引构建3x3旋转矩阵
                    rotation_matrix = np.array([
                        values[0], values[1], values[2],
                        values[4], values[5], values[6],
                        values[8], values[9], values[10]
                    ]).reshape(3, 3)
                    
                    # 使用指定的索引构建平移向量
                    translation_vector = np.array([values[3], values[7], values[11]])
                else:
                    raise ValueError("Tr_velo_to_cam 的值数量不正确，应该是12个")
    return rotation_matrix, translation_vector

def create_bbox(x, y, z, h, w, l, rotation_y, object_type, calibration_matrix, translation_vector):
    """
    创建边界框并应用旋转和平移变换。

    参数：
        x, y, z: 边界框中心的坐标。
        h, w, l: 边界框的高度、宽度和长度。
        rotation_y: 边界框的旋转角度。
        object_type: 对象类型（如行人、车辆等）。
        calibration_matrix: 从激光雷达到相机的旋转矩阵。
        translation_vector: 从激光雷达到相机的平移向量。

    返回：
        bbox: 创建的边界框对象。
    """
    bbox = o3d.geometry.OrientedBoundingBox()
    
    # 设置边界框的中心和尺寸
    if object_type == "Pedestrian":  # 行人
        bbox.center = np.array([x, y, z])  # 使用原始中心
        bbox.extent = [h, w, l]
    elif object_type == "Car":  # 车辆
        bbox.center = np.array([x, y, h / 2 + 0.32])  # 底部中心，z=0
        bbox.extent = [h, w, l]

    # 设置旋转
    R = np.array([
        [np.cos(rotation_y), -np.sin(rotation_y), 0],
        [np.sin(rotation_y), np.cos(rotation_y), 0],
        [0, 0, 1]
    ])
    
    # 使用标定矩阵进行旋转
    R = R @ calibration_matrix[:3, :3]  # 只取旋转部分
    translation_vector =  calibration_matrix @ -translation_vector
    # 应用旋转
    bbox.rotate(R, center=bbox.center)

    # 应用平移
    bbox.translate(translation_vector)  # 将平移向量添加到边界框中心

    return bbox

def visualize(point_cloud, bboxes):
    """
    可视化点云和边界框。

    参数：
        point_cloud: 点云数据。
        bboxes: 边界框列表。
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 设置边界框的颜色
    for bbox in bboxes:
        bbox.color = (1, 0, 0)  # 红色

    # 可视化
    o3d.visualization.draw_geometries([pcd] + bboxes)

if __name__ == "__main__":
    # 定义数据文件夹和文件ID
    data_folder = "data/training"
    file_id = "000026"
    
    # 定义标定文件路径
    calibration_file_path = f"{data_folder}/calib/{file_id}.txt"
    
    # 读取标定数据
    rotation_matrix, translation_vector = read_calibration(calibration_file_path)
    
    # 读取点云数据
    point_cloud = read_point_cloud(f"{data_folder}/velodyne/{file_id}.bin")
    
    # 读取边界框数据
    bboxes = read_bounding_boxes(f"{data_folder}/lidar_label/{file_id}.txt", rotation_matrix, translation_vector)
    
    # 可视化点云和边界框
    visualize(point_cloud, bboxes)
    print("Done")
