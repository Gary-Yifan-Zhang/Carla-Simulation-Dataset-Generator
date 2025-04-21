import yaml
import carla
import math
import numpy as np
import logging
import logging
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re  # 添加这行导入

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

def parse_filename(filename):
    """解析文件名获取帧号和摄像头ID"""
    match = re.match(r"^(\d+)_camera_(\d+)\.txt$", filename)
    if not match:
        raise ValueError(f"文件名格式错误: {filename}")
    return int(match.group(1)), int(match.group(2))


def parse_kitti_line(line):
    """解析单行KITTI标签数据"""
    parts = line.strip().split()
    if len(parts) != 16:
        return None
    
    return {
        "bbox": [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])],
        "confidence": 1.0,  # KITTI标签默认置信度为1
        "class_id": class_name_to_id(parts[0]),
        "class_name": parts[0].lower()
    }

def class_name_to_id(class_name):
    """将类别名称转换为ID"""
    class_map = {"pedestrian": 0, "car": 1, "cyclist": 2}
    return class_map.get(class_name.lower(), -1)

def convert_labels(input_dir, output_dir, fps=24, resolution=(480, 320)):
    """
    主转换函数
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param fps: 视频帧率（默认24）
    :param resolution: 视频分辨率（默认480x320）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 组织摄像头数据
    cameras = {}
    
    # 处理所有标签文件
    for file in tqdm(sorted(input_path.glob("*_camera_*.txt")), desc="处理进度"):
        try:
            frame_num, cam_id = parse_filename(file.name)
            
            # 初始化摄像头数据结构
            if cam_id not in cameras:
                cameras[cam_id] = {
                    "video_info": {
                        "input_path": str(file.resolve()),
                        "output_video": f"results/camera_{cam_id}/annotated_video.mp4",
                        "fps": fps,
                        "total_frames": 0,
                        "resolution": list(resolution)
                    },
                    "detections": []
                }
             # 读取检测数据
            with open(file, 'r') as f:
                detections = [parse_kitti_line(l) for l in f if l.strip()]
                detections = [d for d in detections if d is not None]
            
            # 构建帧数据
            frame_data = {
                "frame_number": frame_num,
                "timestamp": round(frame_num / fps, 3),
                "detections": detections
            }
            
            cameras[cam_id]["detections"].append(frame_data)
            cameras[cam_id]["video_info"]["total_frames"] += 1

        except Exception as e:
            print(f"处理文件 {file.name} 出错: {str(e)}")
            continue

     # 保存每个摄像头数据
    for cam_id, data in cameras.items():
        # 按帧号排序
        data["detections"].sort(key=lambda x: x["frame_number"])
        
        # 补全缺失帧
        full_detections = []
        expected_frame = 0
        for d in data["detections"]:
            while expected_frame < d["frame_number"]:
                full_detections.append({
                    "frame_number": expected_frame,
                    "timestamp": round(expected_frame / fps, 3),
                    "detections": []
                })
                expected_frame += 1
            full_detections.append(d)
            expected_frame += 1
        
        data["detections"] = full_detections
        data["video_info"]["total_frames"] = len(full_detections)

        # 保存文件
        output_file = output_path / f"camera_{cam_id}_annotations.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"摄像头{cam_id}数据已保存至: {output_file}")


if __name__ == "__main__":
    base_dir = './data/training_20250420_153627_DynamicObjectCrossing_9'

    # 基于base路径定义其他路径
    label_dir = f'{base_dir}/image_label'
    new_label_dir = f'{base_dir}/new_view/label'
    output_dir = f'{base_dir}/bbox_labels'
    
    try:
        convert_labels(label_dir, output_dir, fps=24, resolution=(960, 640))
    except Exception as e:
        print(f"发生错误：{str(e)}")