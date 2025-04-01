import yaml
import carla
import math
import numpy as np
import logging
import logging

def yaml_to_config(file):
    """
        从yaml文件中读取config

        参数：
            file：文件路径

        返回：
            config：预设配置
    """
    try:
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except:
        return None


def config_transform_to_carla_transform(config_transform):
    """
        将config中的位姿转换为carla中的位姿

        参数：
            config_transform：config中的位姿

        返回：
            carla_transform：carla中的位姿
    """
    carla_transform = carla.Transform(carla.Location(config_transform["location"][0],
                                                     config_transform["location"][1],
                                                     config_transform["location"][2]),
                                      carla.Rotation(config_transform["rotation"][0],
                                                     config_transform["rotation"][1],
                                                     config_transform["rotation"][2]))
    return carla_transform


def set_camera_intrinsic(width, height):
    """
        设置相机内参矩阵

        参数：
            width：图像宽度(pixel)
            height：图像高度(pixel)

        返回：
            k：相机内参矩阵
    """
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def object_filter_by_distance(data, distance):
    """
        根据预设距离对场景中的物体进行简单过滤

        参数：
            data：CARLA传感器相关数据（原始数据，内参，外参等）
            distance：预设的距离阈值

        返回：
            data：处理后的CARLA传感器相关数据
    """
    environment_objects = data["environment_objects"]
    actors = data["actors"]
    for agent, _ in data["agents_data"].items():
        data["environment_objects"] = [obj for obj in environment_objects if
                                       agent.get_location().distance(obj.transform.location) < distance]
        data["actors"] = [act for act in actors if
                          agent.get_location().distance(act.get_location()) < distance]
    return data


def raw_image_to_rgb_array(image):
    """
        将CARLA原始图像数据转换为RGB numpy数组

        参数：
            image：CARLA原始图像数据

        返回：
            array：RGB numpy数组
    """
    # 将CARLA原始数据转化为BGRA numpy数组
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))

    # 将BGRA numpy数组转化为RGB numpy数组
    # 向量只取BGR三个通道
    array = array[:, :, :3]
    # 倒序
    array = array[:, :, ::-1]
    return array


def depth_image_to_array(image):
    """
        将carla获取的raw depth_image转换成深度图
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Take only RGB
    array = array[:, :, ::-1]  # BGR
    array = array.astype(np.float32)  # 2ms
    gray_depth = ((array[:, :, 0] + array[:, :, 1] * 256.0 + array[:, :, 2] * 256.0 * 256.0) / (
            (256.0 * 256.0 * 256.0) - 1))  # 2.5ms
    gray_depth = 1000 * gray_depth
    return gray_depth

def calculate_extrinsic_matrix(transform):
    """计算4x4齐次变换矩阵
    
    参数：
        transform: carla.Transform对象
        
    返回：
        4x4 numpy数组，包含旋转和平移信息
    """
    # 提取平移分量
    translation = np.array([
        transform.location.x,
        transform.location.y,
        transform.location.z
    ])
    
    # 将欧拉角转换为弧度
    roll = math.radians(transform.rotation.roll)
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    
    # 计算绕Z轴旋转矩阵（yaw）
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 计算绕Y轴旋转矩阵（pitch）
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # 计算绕X轴旋转矩阵（roll）
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # 组合旋转矩阵（ZYX顺序）
    rotation = Rz @ Ry @ Rx
    
    # 构建4x4齐次变换矩阵
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    
    logging.debug(f"Calculated extrinsic matrix for {transform}:\n{extrinsic}")
    return extrinsic

def load_extrinsic_npz(file_path, sensor_name=None):
    """
    加载外参npz文件工具函数
    
    参数：
        file_path: npz文件路径
        sensor_name: 可选，指定要获取的传感器名称
    
    返回：
        若指定sensor_name: 返回对应4x4矩阵
        未指定sensor_name: 返回(extrinsics_dict, sensor_mapping)
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # 修改：处理sensor_mapping的加载
        if 'sensor_mapping' in data:
            sensor_mapping = data['sensor_mapping']
            # 如果sensor_mapping是numpy数组且长度为1，则提取其内容
            if isinstance(sensor_mapping, np.ndarray) and sensor_mapping.size == 1:
                sensor_mapping = sensor_mapping.item()
        else:
            sensor_mapping = {}
        
        # 构建外参字典
        extrinsics = {}
        for key in data.keys():
            if key != 'sensor_mapping':  # 排除sensor_mapping
                extrinsics[key] = data[key]
        
        if sensor_name:
            if sensor_name not in extrinsics:
                raise KeyError(f"传感器 {sensor_name} 不存在于文件中，可用传感器: {list(extrinsics.keys())}")
            logging.info(f"成功加载 {sensor_name} 外参矩阵")
            return extrinsics[sensor_name]
        else:
            logging.info(f"成功加载 {len(extrinsics)} 个外参矩阵")
            return extrinsics, sensor_mapping
            
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 不存在")
        raise
    except Exception as e:
        logging.error(f"加载外参文件失败: {str(e)}")
        raise
    
if __name__ == "__main__":
    # 示例1：加载全部数据
    all_extrinsics, mapping = load_extrinsic_npz("./data/training_20250326_111957/extrinsic/000001.npz")
    print(f"包含传感器: {list(all_extrinsics.keys())}")
    print(f"RGB传感器矩阵:\n{all_extrinsics['RGB']}")
    print(f"LIDAR传感器矩阵:\n{all_extrinsics['LIDAR']}")
    print(f"SUB RGB1传感器矩阵:\n{all_extrinsics['SUB_RGB_1']}")