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
    # print("point cloud shape:", np.fromfile(file_path, dtype=np.float32).shape)
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    print("point cloud shape:", point_cloud.shape)
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
        metadata: 包含边界框元数据的字典
    """
    bboxes = []
    metadata = {}  # 新增元数据存储
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            object_type = data[0]
            h, l, w = float(data[9]), float(data[10]), float(data[11])
            print(f"\n检测到 {object_type} 边界框 - 原始尺寸（高/宽/长）: {h:.2f}m/{w:.2f}m/{l:.2f}m")
            # 将hwl膨胀5%
            h *= 1.05
            w *= 1.05
            l *= 1.05
            x, y, z = float(data[12]), float(data[13]), float(data[14])
            print(object_type,x,y,z)
            rotation_y = float(data[15])

            bbox = create_bbox(x, y, z, h, w, l, rotation_y, object_type, calibration_matrix, translation_vector)
            
            # 存储元数据
            metadata[bbox] = {
                'object_type': object_type,
                'original_center': np.array([x, y, z]),
                'original_extent': [h, w, l]
            }
            
            bboxes.append(bbox)
    return bboxes, metadata  # 修改返回值为元组

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
    创建边界框并应用旋转和平移变换

    参数优化说明：
    1. 使用类型颜色映射字典统一管理颜色配置
    2. 合并同类条件判断
    3. 显式处理z轴偏移量
    4. 添加类型校验和默认处理
    """
    # 类型到颜色的映射配置
    TYPE_COLORS = {
        "Pedestrian": (1, 0, 0),    # 红色-行人
        "Car": (0, 1, 0),          # 绿色-车辆
        "Bicycle": (1, 1, 0),       # 黄色-自行车
        "TrafficLight": (0, 0, 1),  # 蓝色-交通灯
        "TrafficSigns": (0, 0, 1)   # 蓝色-交通标志
    }

    # 初始化边界框
    bbox = o3d.geometry.OrientedBoundingBox()
    
    # 统一设置尺寸（所有类型使用相同尺寸逻辑）
    bbox.extent = [l,w,h]
    
    # 设置中心点高度偏移（大部分类型需要提升h/2）
    z_offset = h / 2 if object_type in ["Pedestrian", "Car", "Bicycle", "TrafficLight", "TrafficSigns"] else 0
    bbox.center = np.array([x, y, z + z_offset])

    # 设置颜色（默认使用蓝色）
    bbox.color = TYPE_COLORS.get(object_type, (0, 0, 1))

    # 构造旋转矩阵
    theta = rotation_y
    R_obj = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,              1]
    ])
    
    
    # 执行旋转变换（保持中心点不变）
    bbox.rotate(R_obj, center=bbox.center)

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
    
     # 创建坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])


    # 可视化
    o3d.visualization.draw_geometries([pcd, axis] + bboxes)


def calculate_bbox_volume(bbox):
    """
    计算边界框的体积

    参数：
        bbox: 边界框对象

    返回：
        volume: 边界框的体积
    """
    # 获取边界框的尺寸
    extent = bbox.extent
    # 计算体积：长 * 宽 * 高
    volume = extent[0] * extent[1] * extent[2]
    return volume

def check_bbox_size(bboxes, metadata, threshold=1.0):
    """
    检查边界框的体积是否小于阈值

    参数：
        bboxes: 边界框列表
        metadata: 边界框元数据字典
        threshold: 体积阈值，默认为1.0

    返回：
        small_bboxes: 体积小于阈值的边界框列表
    """
    small_bboxes = []
    for bbox in bboxes:
        volume = calculate_bbox_volume(bbox)
        if volume < threshold:
            small_bboxes.append(bbox)
            # 从元数据字典获取信息
            info = metadata[bbox]
            print(f"小体积边界框信息：")
            print(f"  类型: {info['object_type']}")
            print(f"  原始中心点: {info['original_center']}")
            print(f"  原始尺寸 (长, 宽, 高): {info['original_extent']}")
            print(f"  变换后尺寸: {bbox.extent}")
            print(f"  体积: {volume:.2f}")
            
            # 调整边界框尺寸
            original_extent = bbox.extent
            bbox.extent = [original_extent[0],  # 高度不变
                          original_extent[1] + 0.3,  # 宽度增加0.5
                          original_extent[2] + 0.3]  # 长度增加0.5
            print(f"  调整后尺寸: {bbox.extent}")
            print("-" * 30)
    return small_bboxes

if __name__ == "__main__":
    # 定义数据文件夹和文件ID
    data_folder = "data/training_20250324_142316"
    file_id = "000002"
    file_id_1 = "000010"
    lidar_index = 0  # 假设这是第一个雷达数据
    
    # 定义标定文件路径
    calibration_file_path = f"{data_folder}/calib/{file_id}.txt"
    
    # 读取标定数据
    rotation_matrix, translation_vector = read_calibration(calibration_file_path)
    
    # 读取点云数据，使用新的命名格式
    point_cloud = read_point_cloud(f"{data_folder}/velodyne/{file_id}_lidar_{lidar_index}.bin")
    print(f"点云数据目录: {data_folder}/velodyne/{file_id}_lidar_{lidar_index}.bin")
    
    # 读取边界框数据（修改接收方式）
    bboxes, bbox_metadata = read_bounding_boxes(f"{data_folder}/lidar_label/{file_id}.txt", rotation_matrix, translation_vector)
    
    # 检查小体积的边界框（添加metadata参数）
    threshold = 1.0
    small_bboxes = check_bbox_size(bboxes, bbox_metadata, threshold)
    print(f"找到 {len(small_bboxes)} 个体积小于 {threshold} 的边界框")
    
    # 可视化点云和边界框
    visualize(point_cloud, bboxes)
    print("Done")
