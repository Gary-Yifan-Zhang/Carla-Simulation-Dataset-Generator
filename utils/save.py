import numpy as np
from PIL import Image
import os
import logging
import math
import carla
from utils.utils import config_transform_to_carla_transform, calculate_extrinsic_matrix


def save_ref_files(folder, index):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(folder, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(index) + '\n')
        logging.info("Wrote reference files to %s", path)


def save_image_data(filename, image):
    """
        保存RGB图像

        参数：
            filename：保存文件的路径
            image：CARLA原始图像数据
    """
    logging.info("Wrote image data to %s", filename)
    image.save_to_disk(filename)
    
def save_semantic_image_data(filename, semantic_image):
    """
    保存语义分割图像数据

    参数：
        filename: 保存文件的完整路径（包括文件名和扩展名）
        semantic_image: 从CARLA获取的语义分割图像数据

    说明：
        1. 使用CityScapes调色板保存语义分割图像
        2. 输出文件格式为PNG
        3. 图像包含不同类别的颜色编码信息
    """
    logging.info("Wrote semantic image data to %s", filename)
    semantic_image.save_to_disk(filename, carla.ColorConverter.CityScapesPalette)


def save_depth_image_data(filename, depth_image):
    """
        保存深度图像

        参数：
            filename：保存文件的路径
            depth_image：CARLA原始深度图像数据
    """
    logging.info("Wrote depth image data to %s", filename)
    depth_image.save_to_disk(filename, carla.ColorConverter.LogarithmicDepth)


def save_bbox_image_data(filename, image):
    """
        保存带有2d bounding box的RGB图像

        参数：
            filename：保存文件的路径
            image：带有2d bounding box的RGB图像数组
    """
    img = Image.fromarray(image)
    img.save(filename)


def save_lidar_data(filename, point_cloud, extrinsic, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right-handed coordinate system
        Therefore, we need to flip the y-axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        point_cloud = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
        point_cloud = point_cloud[:, :-1]

        lidar_array = [[point[0], point[1], point[2], 1.0] for point in point_cloud]
        lidar_array = np.dot(extrinsic, np.array(lidar_array).transpose())
        # 减掉因将激光雷达为原点而导致的偏移量
        lidar_array[0, :] = lidar_array[0, :] - extrinsic[0, 3]
        lidar_array[1, :] = -(lidar_array[1, :] - extrinsic[1, 3])
        lidar_array = lidar_array.transpose().astype(np.float32)

        logging.debug("Lidar min/max of x: {} {}".format(
            lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of y: {} {}".format(
            lidar_array[:, 1].min(), lidar_array[:, 1].max()))
        logging.debug("Lidar min/max of z: {} {}".format(
            lidar_array[:, 2].min(), lidar_array[:, 2].max()))
        lidar_array.tofile(filename)


def save_semantic_lidar_data(filename, data):
    """
        保存CARLA语义激光雷达数据

        参数：
            filename：保存文件的路径
            data：CARLA语义激光雷达数据
    """
    sem_lidar = np.frombuffer(data.raw_data, dtype=np.dtype('f4,f4, f4, f4, i4, i4'))
    np.savetxt(filename, sem_lidar, fmt="%.4f %.4f %.4f %.4f %d %d")


def save_kitti_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)


def save_calibration_matrices(transform, filename, intrinsic_mat):
    """
        保存传感器标定矩阵数据

        参数：
            transform：相机和激光雷达外参矩阵
            filename：保存文件的路径
            intrinsic_mat：RGB相机内参矩阵

        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        保存的文件中包含:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect              相机畸变矩阵
        3x4    tr_velodyne_to_cam   激光雷达坐标系到相机坐标系的变换矩阵
                                    Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne.

        3x4    tr_imu_to_velo       IMU坐标系到激光雷达坐标系的变换矩阵（此处无IMU，故不输出）
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)

    camera_transform = transform[0]
    lidar_transform = transform[1]

    # 提取平移信息
    camera_translation = np.array([camera_transform.location.x, camera_transform.location.y, camera_transform.location.z])
    lidar_translation = np.array([lidar_transform.location.x, lidar_transform.location.y, lidar_transform.location.z])

    # 计算旋转角度
    b = math.radians(lidar_transform.rotation.pitch - camera_transform.rotation.pitch)
    x = math.radians(lidar_transform.rotation.yaw - camera_transform.rotation.yaw)
    a = math.radians(lidar_transform.rotation.roll - camera_transform.rotation.roll)
    R0 = np.identity(3)

    # 计算旋转矩阵
    TR = np.array([[math.cos(b) * math.cos(x), math.cos(b) * math.sin(x), -math.sin(b)],
                   [-math.cos(a) * math.sin(x) + math.sin(a) * math.sin(b) * math.cos(x),
                    math.cos(a) * math.cos(x) + math.sin(a) * math.sin(b) * math.sin(x), math.sin(a) * math.cos(b)],
                   [math.sin(a) * math.sin(x) + math.cos(a) * math.sin(b) * math.cos(x),
                    -math.sin(a) * math.cos(x) + math.cos(a) * math.sin(b) * math.sin(x), math.cos(a) * math.cos(b)]])

    # 计算激光雷达到相机的变换矩阵
    TR_velodyne = np.dot(TR, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))
    TR_velodyne = np.dot(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]), TR_velodyne)

    # 添加平移向量
    translation = lidar_translation - camera_translation  # 计算平移向量
    TR_velodyne = np.column_stack((TR_velodyne, translation))  # 将平移向量添加到变换矩阵中

    # 处理IMU到激光雷达的变换矩阵
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    # 保存矩阵到文件
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)


def write_flat(file, name, arr):
    ravel_mode = 'C'
    file.write("{}: {}\n".format(name, ' '.join(
        map(str, arr.flatten(ravel_mode).squeeze()))))


def save_ego_data(filename, transform, rotation, velocity, acceleration, extent):
    """
    增强版ego状态保存，包含完整信息
    """
    with open(filename, 'w') as f:
        f.write("=== Ego State ===\n")
        f.write(f"Location (x,y,z): {transform['x']:.3f}, {transform['y']:.3f}, {transform['z']:.3f}\n")
        f.write(f"Rotation (yaw,pitch,roll): {rotation['yaw']:.2f}, {rotation['pitch']:.2f}, {rotation['roll']:.2f}\n")
        f.write(f"Velocity (m/s): {velocity['x']:.2f}, {velocity['y']:.2f}, {velocity['z']:.2f}\n")
        f.write(f"Acceleration (m/s²): {acceleration['x']:.2f}, {acceleration['y']:.2f}, {acceleration['z']:.2f}\n")
        f.write(f"Bounding Box (L,W,H): {extent['x']:.2f}m, {extent['y']:.2f}m, {extent['z']:.2f}m\n")
    
    logging.info("Updated ego state saved to %s", filename)
    
def save_semantic_image(filename, semantic_image):
    """ 保存语义分割图像到指定文件

    参数：
        filename：保存文件的路径
        semantic_image：语义分割图像数组，通常是单通道的类别标签
    """
    # 将语义图像转换为PIL图像
    img = Image.fromarray(semantic_image.astype(np.uint8))
    # 保存图像
    img.save(filename)



def save_extrinsic_matrices(config, base_filename, sensor_mapping):
    """
    按传感器映射保存单独的外参文件
    
    参数：
        config: 配置文件对象
        base_filename: 基础文件名格式（需包含{id}占位符）
        sensor_mapping: 传感器到文件编号的映射字典
    """
    
    for sensor_name, file_id in sensor_mapping.items():
        filename = base_filename.format(id=file_id)
        
        # 从配置获取传感器transform并转换为carla.Transform
        transform = config_transform_to_carla_transform(
            config["SENSOR_CONFIG"][sensor_name]["TRANSFORM"]
        )
        
        # 构建齐次变换矩阵
        translation = np.array([
            transform.location.x,
            transform.location.y, 
            transform.location.z
        ])
        
        # 将欧拉角转换为旋转矩阵
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)
        
        # 旋转矩阵（ZYX顺序）
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        rotation = Rz @ Ry @ Rx
        # 构建4x4齐次矩阵
        extrinsic = np.identity(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        
        with open(filename, 'w') as f:     
            np.savetxt(f, extrinsic, fmt='%.6f')
            
        logging.info("Wrote %s extrinsic matrix to %s", sensor_name, filename)

def save_globel_extrinsic_matrices(filename, sensor_mapping, extrinsic_dict):
    """
    保存所有传感器的外参矩阵到单个文件
    
    参数：
        filename: 输出文件名
        sensor_mapping: 传感器到索引的映射字典
        extrinsic_dict: 包含各传感器4x4外参矩阵的字典
    """
    processed_dict = {}
    
    for sensor_name, matrix in extrinsic_dict.items():
        # 转换numpy矩阵格式
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
            
        # 验证矩阵维度
        if matrix.shape != (4, 4):
            raise ValueError(f"{sensor_name} 外参矩阵维度错误，应为4x4，实际为{matrix.shape}")
            
        processed_dict[sensor_name] = matrix.astype(np.float32)
    
    # 添加传感器映射关系
    processed_dict["sensor_mapping"] = np.array(sensor_mapping)
    
    # 保存为numpy压缩格式
    np.savez_compressed(filename, **processed_dict)
    logging.info(f"已保存外参矩阵至 {filename}")
    
def save_extrinsic_txt(config, filename, sensor_mapping):
    """
    保存标准txt格式外参（每帧一个文件）
    格式示例：
    RGB
    0.999 0.012 0.034 1.234
    0.011 0.998 0.052 2.345
    0.033 0.051 0.997 3.456
    0.000 0.000 0.000 1.000
    LIDAR
    0.888 0.023 0.456 4.567
    ...
    """
    with open(filename, 'w') as f:
        for sensor_name in sensor_mapping.keys():
            # 获取变换矩阵
            transform = config_transform_to_carla_transform(
                config["SENSOR_CONFIG"][sensor_name]["TRANSFORM"]
            )
            
            # 计算4x4齐次矩阵（与之前相同）
            extrinsic = calculate_extrinsic_matrix(transform)  # 复用原有矩阵计算逻辑
            
            # 写入传感器名称
            f.write(f"{sensor_name}\n")
            
            # 写入矩阵数据（保留6位小数）
            for row in extrinsic:
                f.write(" ".join([f"{x:.6f}" for x in row]) + "\n")
            f.write("\n")  # 添加空行分隔不同传感器
    
    logging.info(f"Saved txt extrinsic to {filename}")