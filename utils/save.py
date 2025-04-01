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
        lidar_array[0, :] = (lidar_array[0, :] - extrinsic[0, 3])
        lidar_array[1, :] = -(lidar_array[1, :] - extrinsic[1, 3])
        lidar_array[2, :] = (lidar_array[2, :] - extrinsic[2, 3])
        # 依然是雷达坐标系，转换为右手系，即x轴不变，y轴取反，z轴不变
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


def save_calibration_matrices(config, filename, sensor_mapping, intrinsic_mat):
    """
        保存传感器标定矩阵数据（支持特定索引的相机和雷达）

        参数：
            extrinsics：所有传感器的外参矩阵列表
            filename：保存文件的路径
            intrinsic_mat：相机的内参矩阵

        保存的文件中包含:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect              相机畸变矩阵
        3x4    tr_velo_to_cam       激光雷达坐标系到相机坐标系的变换矩阵
        3x4    TR_imu_to_velo       IMU到激光雷达的变换矩阵
    """
    # KITTI format demands that we flatten in row-major order
     # 自动识别雷达和摄像头
    lidar_sensor = [k for k in sensor_mapping if "LIDAR" in k.upper()][0]
    camera_sensors = [k for k in sensor_mapping if k != lidar_sensor]

    # 获取雷达外参
    lidar_transform = config_transform_to_carla_transform(
        config["SENSOR_CONFIG"][lidar_sensor]["TRANSFORM"]
    )
    lidar_extrinsic = calculate_extrinsic_matrix(lidar_transform)

    with open(filename, 'w') as f:
        # 写入P矩阵（所有P矩阵相同）
        P0 = np.column_stack((intrinsic_mat, np.zeros(3)))
        P0_flat = P0.ravel(order='C')
        for i in range(4):
            f.write(f"P{i}: {' '.join(f'{x:.6f}' for x in P0_flat)}\n")

        # 写入R0_rect（单位矩阵）
        f.write(f"R0_rect: {' '.join('1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000')}\n")

        # 为每个摄像头生成变换矩阵
        for idx, cam_sensor in enumerate(camera_sensors):
            # 获取相机外参
            cam_transform = config_transform_to_carla_transform(
                config["SENSOR_CONFIG"][cam_sensor]["TRANSFORM"]
            )
            # 获取完整的4x4外参矩阵
            cam_extrinsic = calculate_extrinsic_matrix(cam_transform)
            lidar_extrinsic = calculate_extrinsic_matrix(lidar_transform)

            # 计算雷达外参的逆矩阵
            lidar_extrinsic_inv = np.linalg.inv(lidar_extrinsic)

            # 计算雷达到相机的变换矩阵（CARLA坐标系）
            lidar_to_camera = cam_extrinsic @ lidar_extrinsic_inv

            # 应用坐标系转换矩阵（CARLA到KITTI）
            velo_to_cam = np.array([
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            TR_adjusted = velo_to_cam @ lidar_to_camera

            # 提取前3行（3x4矩阵）
            TR = TR_adjusted[:3, :]
            
            # 写入调整后的矩阵
            tr_values = ' '.join(f"{x:.6f}" for x in TR.ravel(order='C'))
            f.write(f"Tr_velo_to_cam_{idx}: {tr_values}\n")

        # IMU到雷达的变换（单位矩阵）
        f.write("TR_imu_to_velo: 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000\n")

    logging.info(f"Calibration matrices saved to {filename}")


def write_flat(file, name, arr):
    ravel_mode = 'C'
    file.write("{}: {}\n".format(name, ' '.join(
        map(str, arr.flatten(ravel_mode).squeeze()))))


def save_ego_data(filename, transform, rotation, velocity, acceleration, extent):
    """
    增强版ego状态保存，包含完整信息
    """
    with open(filename, 'w') as f:
        # 写入Transform
        f.write(f"Transform: Transform(Location(x={transform['x']:.6f}, y={transform['y']:.6f}, z={transform['z']:.6f}), "
                f"Rotation(pitch={rotation['pitch']:.6f}, yaw={rotation['yaw']:.6f}, roll={rotation['roll']:.6f}))\n")
        # 写入Velocity
        f.write(f"Velocity: {{'x': {velocity['x']:.1f}, 'y': {velocity['y']:.1f}, 'z': {velocity['z']:.6f}}}\n")
        # 写入Acceleration
        f.write(f"Acceleration: {{'x': {acceleration['x']:.1f}, 'y': {acceleration['y']:.1f}, 'z': {acceleration['z']:.6f}}}\n")
    
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