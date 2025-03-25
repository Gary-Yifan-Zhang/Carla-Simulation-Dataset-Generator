import carla
import math
from utils.utils import raw_image_to_rgb_array, yaml_to_config, depth_image_to_array
from KittiDescriptor import KittiDescriptor
from utils.visual import *
from utils.transform import *

config = yaml_to_config("configs.yaml")
MAX_RENDER_DEPTH_IN_METERS = config["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = config["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = config["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
MIN_VISIBLE_NUM_FOR_POINT_CLOUDS = config["FILTER_CONFIG"]["MIN_VISIBLE_NUM_FOR_POINT_CLOUDS"]
WINDOW_WIDTH = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]


GLOBAL_ID_MAP = {}
NEXT_ID = 0


def spawn_dataset(data):
    """
        处理传感器原始数据，生成KITTI数据集(RGB图像，激光雷达点云，KITTI标签等)

        参数：
            data：CARLA传感器相关数据（原始数据，内参，外参等）

        返回：
            data：处理后的数据（RGB图像，激光雷达点云，KITTI标签等）
    """
    # 筛选环境中的车辆
    environment_objects = data["environment_objects"]
    environment_objects = [x for x in environment_objects if x.type == "vehicle" or x.type in [carla.CityObjectLabel.TrafficSigns, 
                                      carla.CityObjectLabel.TrafficLight]]

    # 筛选actors中的车辆与行人
    actors = data["actors"]
    actors = [x for x in actors if x.type_id.find("vehicle") != -1 or x.type_id.find("walker") != -1]

    agents_data = data["agents_data"]

    
    for agent, dataDict in agents_data.items():
        GLOBAL_ID_MAP[agent.id] = 0
        intrinsic = dataDict["intrinsic"]
        transform = dataDict["transform"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        ego_state = data["egostate"]

        image_labels_kitti = []
        image_labels_kitti_1 = []
        image_labels_kitti_2 = []
        pc_labels_kitti = []

        rgb_image = raw_image_to_rgb_array(sensors_data[0])
        rgb_image_1 = raw_image_to_rgb_array(sensors_data[4])
        rgb_image_2 = raw_image_to_rgb_array(sensors_data[5])
        rgb_image_3 = raw_image_to_rgb_array(sensors_data[13])
        rgb_image_4 = raw_image_to_rgb_array(sensors_data[14])
        image = rgb_image.copy()
        image_1 = rgb_image_1.copy()
        image_2 = rgb_image_2.copy()
        image_3 = rgb_image_3.copy()
        image_4 = rgb_image_4.copy()


        depth_data = depth_image_to_array(sensors_data[1])
        depth_data_1 = depth_image_to_array(sensors_data[11])
        depth_data_2 = depth_image_to_array(sensors_data[12])
        depth_data_3 = depth_image_to_array(sensors_data[17])
        depth_data_4 = depth_image_to_array(sensors_data[18])
        semantic_lidar = np.frombuffer(sensors_data[3].raw_data, dtype=np.dtype('f4,f4, f4, f4, i4, i4'))

        # 对环境中的目标物体生成标签
        data["agents_data"][agent]["visible_environment_objects"] = []
        for obj in environment_objects:
            image_label_kitti = is_visible_in_camera(agent, obj, image, depth_data, intrinsic, transform[0])
            if image_label_kitti is not None:
                data["agents_data"][agent]["visible_environment_objects"].append(obj)
                image_labels_kitti.append(image_label_kitti)

            pc_label_kitti = is_visible_in_lidar(agent, obj, semantic_lidar, extrinsic[0], ego_state)
            if pc_label_kitti is not None:
                pc_labels_kitti.append(pc_label_kitti)

        # 对actors中的目标物体生成标签
        data["agents_data"][agent]["visible_actors"] = []
        for act in actors:
            image_label_kitti = is_visible_in_camera(agent, act, image, depth_data, intrinsic, transform[0])
            if image_label_kitti is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                image_labels_kitti.append(image_label_kitti)

            pc_label_kitti = is_visible_in_lidar(agent, act, semantic_lidar, extrinsic[0], ego_state)
            if pc_label_kitti is not None:
                pc_labels_kitti.append(pc_label_kitti)
                
        # 新增对第二个摄像头（sensors_data[4]）的处理
        image_labels_kitti_1 = process_camera_view(
            agent, 
            environment_objects + actors,  # 合并环境和演员对象
            image_1, 
            depth_data_1,
            intrinsic,
            transform[4]
        )

        # 处理第二个摄像头（transform[5]）
        image_labels_kitti_2 = process_camera_view(
            agent,
            environment_objects + actors,
            image_2,
            depth_data_2,
            intrinsic,
            transform[5]
        )
        
        image_labels_kitti_3 = process_camera_view(
            agent, 
            environment_objects + actors,  # 合并环境和演员对象
            image_3, 
            depth_data_3,
            intrinsic,
            transform[13]
        )

        # 处理第二个摄像头（transform[5]）
        image_labels_kitti_4 = process_camera_view(
            agent,
            environment_objects + actors,
            image_4,
            depth_data_4,
            intrinsic,
            transform[14]
        )


        data["agents_data"][agent]["rgb_image"] = rgb_image
        data["agents_data"][agent]["bbox_img"] = image
        data["agents_data"][agent]["bbox_img_1"] = image_1   # 新增第二个摄像头标注图
        data["agents_data"][agent]["bbox_img_2"] = image_2   # 新增第三个摄像头标注图
        data["agents_data"][agent]["bbox_img_3"] = image_3   
        data["agents_data"][agent]["bbox_img_4"] = image_4
        data["agents_data"][agent]["image_labels_kitti"] = image_labels_kitti
        data["agents_data"][agent]["image_labels_kitti_1"] = image_labels_kitti_1
        data["agents_data"][agent]["image_labels_kitti_2"] = image_labels_kitti_2
        data["agents_data"][agent]["image_labels_kitti_3"] = image_labels_kitti_3
        data["agents_data"][agent]["image_labels_kitti_4"] = image_labels_kitti_4
        data["agents_data"][agent]["pc_labels_kitti"] = pc_labels_kitti
    return data

def process_camera_view(agent, objects, image, depth_data, intrinsic, transform):
    """处理单个摄像头视角的物体可见性检测"""
    labels = []
    for obj in objects:
        label = is_visible_in_camera(agent, obj, image, depth_data, intrinsic, transform)
        if label is not None:
            labels.append(label)
    return labels

def get_bounding_box(actor, min_extent=0.5, is_environment_object=False):
    """
    修复两轮车辆边界框异常问题
    参数：
        actor: CARLA中的Actor对象
        min_extent: 最小边界框尺寸，默认为0.5米
        is_environment_object: 是否为环境物体，默认为False
    返回：
        carla.BoundingBox: 修正后的边界框
    """
    if not hasattr(actor, "bounding_box"):
        return carla.BoundingBox(carla.Location(0, 0, min_extent), 
                               carla.Vector3D(x=min_extent, y=min_extent, z=min_extent))
    
    bbox = actor.bounding_box
    # 修复Carla 9.11+版本两轮车辆边界框错误
    if bbox.extent.x * bbox.extent.y * bbox.extent.z == 0:
        loc = carla.Location(bbox.extent)
        bbox.extent = carla.Vector3D(bbox.location)
        bbox.location = loc
    
    # 如果是环境物体，跳过以下调整逻辑
    if not is_environment_object:
        # 新增两轮车高度调整逻辑
        if 'crossbike' in actor.type_id or 'motorcycle' in actor.type_id:
            # 调整高度（z轴方向）
            new_height = bbox.extent.z + 0.3
            bbox.extent = carla.Vector3D(
                x=bbox.extent.x,
                y=bbox.extent.y,
                z=new_height
            )
            # 同时调整位置中心点
            bbox.location.z += 0.15  # 高度增加0.4，中心点上移0.2
        
        if 'vehicle' in actor.type_id:
            bbox.extent.x *=1.05
            bbox.extent.y *=1.05
            bbox.extent.z *=1.05
    
    return bbox


def is_visible_in_camera(agent, obj, rgb_image, depth_data, intrinsic, extrinsic):
    """
        筛选出在摄像头中可见的目标物并生成标签

        参数：
            agent：CARLA中agent
            obj：CARLA内物体
            rgb_image：RGB图像
            depth_data：深度信息
            intrinsic：相机内参
            extrinsic：相机外参

        返回：
            kitti_label：包含以下信息的KITTI标签（按照KITTI标准格式顺序）：
                - type: 物体类型（如'Car', 'Pedestrian'等）
                - id: 物体唯一ID，未指定时为-1
                - truncated: 截断程度，0（未截断）到1（完全截断），表示物体离开图像边界的程度
                - occlusion: 遮挡状态整数(0,1,2):
                    0 = 完全可见, 1 = 部分遮挡, 2 = 大部分遮挡
                - alpha: 观测角度，固定为0（相机数据无此信息）
                - bbox: 图像中物体的2D边界框（基于0的索引）:
                    包含左、上、右、下像素坐标
                - dimensions: 3D物体尺寸（高度、宽度、长度）
                - location: 物体在相机坐标系中的3D位置（x, y, z）
                - rotation_y: 物体绕Y轴的旋转角度
    """
    obj_transform = obj.transform if isinstance(obj, carla.EnvironmentObject) else obj.get_transform()
    obj_bbox = get_bounding_box(obj, is_environment_object=isinstance(obj, carla.EnvironmentObject))

    if isinstance(obj, carla.EnvironmentObject):
        vertices_pixel = get_vertices_pixel(intrinsic, extrinsic, obj_bbox, obj_transform, 0)
    else:
        vertices_pixel = get_vertices_pixel(intrinsic, extrinsic, obj_bbox, obj_transform, 1)

    num_visible_vertices, num_vertices_outside_camera = get_occlusion_stats(vertices_pixel, depth_data)
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and \
            num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER:
        if obj.id == agent.id:
            return None
        obj_tp = obj_type(obj)
        rotation_y = get_relative_rotation_yaw(agent.get_transform().rotation, obj_transform.rotation) % math.pi
        midpoint = midpoint_from_world_to_camera(obj_transform.location, extrinsic)
        bbox_2d = get_2d_bbox_in_pixel(vertices_pixel)
        ext = obj_bbox.extent
        truncated = num_vertices_outside_camera / 8
        if num_visible_vertices >= 6:
            occluded = 0
        elif num_visible_vertices >= 4:
            occluded = 1
        else:
            occluded = 2

        # draw_3d_bounding_box(rgb_image, vertices_pos2d)
        bbox_2d_rectified = rectify_2d_bounding_box(bbox_2d)
        draw_2d_bounding_box(rgb_image, bbox_2d_rectified)

        kitti_label = KittiDescriptor()
        kitti_label.set_truncated(truncated)
        kitti_label.set_occlusion(occluded)
        kitti_label.set_bbox(bbox_2d)
        kitti_label.set_3d_object_dimensions(ext)
        kitti_label.set_type(obj_tp)
        kitti_label.set_3d_object_location(midpoint)
        kitti_label.set_rotation_y(rotation_y)
        kitti_label.set_id(get_custom_id(obj, agent))
        return kitti_label
    return None


def is_visible_in_lidar(agent, obj, semantic_lidar, extrinsic, ego_state):
    """
        筛选出在激光雷达中可见的目标物并生成标签

        参数：
            agent：CARLA中agent
            obj：CARLA内物体
            semantic_lidar：语义激光雷达信息（生成的xyz与激光雷达一样，增加了点云所属物体的种类与id）
            extrinsic：激光雷达外参

        返回：
            kitti_label：包含以下信息的KITTI标签（按照KITTI标准格式顺序）：
                - type: 物体类型（如'Car', 'Pedestrian'等）
                - id: 物体唯一ID，未指定时为-1
                - truncated: 截断程度，固定为0（激光雷达不受图像边界限制）
                - occlusion: 遮挡状态，固定为0（激光雷达直接检测物体表面）
                - alpha: 观测角度，固定为0（激光雷达无此信息）
                - bbox: 2D边界框，固定为[0, 0, 0, 0]（激光雷达无2D边界框信息）
                - dimensions: 3D物体尺寸（高度、宽度、长度）
                - location: 物体在激光雷达坐标系中的3D位置（x, y, z）
                - rotation_y: 物体绕Y轴的旋转角度
    """
    pc_num = 0

    # 如果是环境物体，直接检查距离
    if isinstance(obj, carla.EnvironmentObject):
        obj_transform = obj.transform
        distance = math.sqrt(
            (obj_transform.location.x - extrinsic[0, 3])**2 +
            (obj_transform.location.y - extrinsic[1, 3])**2
        )
        if distance <= MAX_RENDER_DEPTH_IN_METERS:
            return create_point_cloud_label(obj, obj_transform, extrinsic, agent, ego_state)
        return None

    # 对于非环境物体，仍然使用点云数量判断
    for point in semantic_lidar:
        if point[4] == obj.id:
            pc_num += 1
        if pc_num >= MIN_VISIBLE_NUM_FOR_POINT_CLOUDS:
            obj_transform = obj.get_transform()
            return create_point_cloud_label(obj, obj_transform, extrinsic, agent, ego_state)
    return None

def create_point_cloud_label(obj, obj_transform, extrinsic, agent, ego_state):
    """
        创建点云标签的通用函数
    """
    if obj.id == agent.id:
            return None
    obj_tp = obj_type(obj)
    # 获取ego的变换矩阵
    ego_matrix = np.array(ego_state["matrix"])
    
    # 物体世界坐标系变换矩阵
    obj_matrix = obj_transform.get_matrix()  # 需要确认Carla是否提供get_matrix方法
    
    # print(extrinsic)
    # 计算相对变换矩阵
    relative_matrix = np.dot(np.linalg.inv(ego_matrix), obj_matrix)
    
     # 增加一步：将相对坐标从ego坐标系转换到雷达坐标系
    lidar_matrix = np.array(extrinsic)  # 雷达外参矩阵
    relative_matrix = np.dot(np.linalg.inv(lidar_matrix), relative_matrix)
    
    
    # 提取位置（直接取变换矩阵的平移分量）
    midpoint = relative_matrix[:3, 3]  # [x, y, z]
    
    # 从相对矩阵提取yaw角
    rotation_y = np.arctan2(relative_matrix[1, 0], relative_matrix[0, 0])
    
    # 转换为KITTI右手坐标系（反转yaw角）
    rotation_y = -rotation_y
    
    
    # 规范化角度到[-π, π]
    rotation_y = np.arctan2(np.sin(rotation_y), np.cos(rotation_y))

    bbox  = get_bounding_box(obj, is_environment_object=isinstance(obj, carla.EnvironmentObject))
    ext = bbox.extent
    point_cloud_label = KittiDescriptor()
    point_cloud_label.set_id(get_custom_id(obj, agent))
    point_cloud_label.set_truncated(0)
    point_cloud_label.set_occlusion(0)
    point_cloud_label.set_bbox([0, 0, 0, 0])
    point_cloud_label.set_3d_object_dimensions(ext)
    point_cloud_label.set_type(obj_tp)
    point_cloud_label.set_lidar_object_location(midpoint)
    point_cloud_label.set_rotation_y(rotation_y)
    return point_cloud_label

def get_occlusion_stats(vertices, depth_image):
    """
        筛选3D bounding box八个顶点在图片中实际可见的点

        参数：
            vertices：物体的3D bounding box八个顶点的像素坐标与深度
            depth_image：深度图片中的深度信息

        返回：
            num_visible_vertices：在图片中可见的bounding box顶点
            num_vertices_outside_camera：在图片中不可见的bounding box顶点
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices:
        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def point_is_occluded(point, vertex_depth, depth_image):
    """
        判断该点是否被遮挡

        参数：
            point：点的像素坐标
            vertex_depth：该点的实际深度
            depth_image：深度图片中的深度信息

        返回：
            bool：是否被遮挡。若是，则返回1;反之则返回0
    """
    y, x = map(int, point)
    from itertools import product
    neighbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neighbours:
        if point_in_canvas((dy + y, dx + x)):
            # 判断点到图像的距离是否大于深对应深度图像的深度值
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # 当四个邻居点都大于深度图像值时，点被遮挡。返回true
    return all(is_occluded)


def get_relative_rotation_yaw(agent_rotation, obj_rotation):
    """
        得到agent和物体在yaw的相对角度

        参数：
            obj：CARLA物体

        返回：
            obj.type：CARLA物体种类
    """
    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return math.radians(rot_agent - rot_car)


def obj_type(obj):
    """
        得到CARLA物体种类，对行人和汽车的种类重命名

        参数：
            obj：CARLA物体

        返回：
            obj.type：CARLA物体种类
    """
    if isinstance(obj, carla.EnvironmentObject):
        return obj.type
    else:
        if obj.type_id.find('walker') != -1:
            return 'Pedestrian'
        if obj.type_id.find('vehicle') != -1:
            if obj.type_id.find('crossbike') != -1:
                return 'Bicycle'
            return 'Car'
            
        return None

def get_custom_id(obj, agent):
    """
    更新后的自定义ID生成逻辑：
    1. 环境物体固定返回-1
    2. ego车辆返回0
    3. 其他物体使用递增ID
    """
    global NEXT_ID, GLOBAL_ID_MAP
    
    # 如果是环境物体
    if isinstance(obj, carla.EnvironmentObject):
        return -1
    
    # 如果是ego车辆
    # if obj.id == agent.id:
    #     return 0
    
    # 如果物体已有映射ID则返回
    if obj.id in GLOBAL_ID_MAP:
        return GLOBAL_ID_MAP[obj.id]
    
    # 为新物体分配ID并递增
    GLOBAL_ID_MAP[obj.id] = NEXT_ID
    NEXT_ID += 1
    return GLOBAL_ID_MAP[obj.id]