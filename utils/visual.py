"""
可视化工具模块

使用说明：
1. 从项目根目录运行：
   python -m utils.visual
2. 或者直接运行：
   python utils/visual.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import yaml_to_config
import open3d as o3d
import cv2
import carla
import math
import numpy as np


config = yaml_to_config("configs.yaml")
WINDOW_WIDTH = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood_fill
from tqdm import tqdm

# 定义类别和颜色映射
ID_TO_COLOR = {
    11: (70, 130, 180),    # Sky
    12: (220, 20, 60),     # Pedestrian
    13: (255, 0, 0),       # Rider
    14: (0, 0, 142),       # Car
    15: (0, 0, 70),        # Truck
    16: (0, 60, 100),      # Bus
    17: (0, 80, 100),      # Train
    18: (0, 0, 230),       # Motorcycle
    19: (119, 11, 32)      # Bicycle
}

# 定义类别分组
SKY_IDS = [11]
RIGID_IDS = [14, 15, 16, 17]
NONRIGID_IDS = [12, 13, 18, 19]


def rectify_2d_bounding_box(bbox_2d):
    """
        对2d bounding box进行校正，将超出图片范围外的部分

            参数：
                bbox_2d：2d bounding box的左上和右下的像素坐标

            返回：
                bbox_2d_rectified：校正后的2d bounding box的左上和右下的像素坐标
    """
    min_x, min_y, max_x, max_y = bbox_2d
    if point_in_canvas((min_y, min_x)) or point_in_canvas((max_y, max_x)) or point_in_canvas((max_y, min_x)) \
            or point_in_canvas((min_y, max_x)):
        min_x_rectified, min_y_rectified = set_point_in_canvas((min_x, min_y))
        max_x_rectified, max_y_rectified = set_point_in_canvas((max_x, max_y))
        bbox_2d_rectified = [min_x_rectified, min_y_rectified, max_x_rectified, max_y_rectified]
        return bbox_2d_rectified
    else:
        return None


def draw_2d_bounding_box(image, bbox_2d):
    """
        在图像中绘制2d bounding box

            参数：
                image：RGB图像
                bbox_2d：2d bounding box的左上和右下的像素坐标

    """
    min_x, min_y, max_x, max_y = bbox_2d

    # 将2d bounding box的四条边设置为红色
    for y in range(min_y, max_y):
        image[y, min_x] = (255, 0, 0)
        image[y, max_x] = (255, 0, 0)

    for x in range(min_x, max_x):
        image[max_y, int(x)] = (255, 0, 0)
        image[min_y, int(x)] = (255, 0, 0)

    # 将2d bounding box的四个顶点设置为绿色
    image[min_y, min_x] = (0, 255, 0)
    image[min_y, max_x] = (0, 255, 0)
    image[max_y, min_x] = (0, 255, 0)
    image[max_y, max_x] = (0, 255, 0)


def draw_3d_bounding_box(image, vertices_2d):
    """
        在图像中绘制3d bounding box

            参数：
                image：RGB图像
                vertices_2d：3d bounding box的8个顶点的像素坐标

    """
    # Shows which verticies that are connected so that we can draw lines between them
    # The key of the dictionary is the index in the bbox array, and the corresponding value is a list of indices
    # referring to the same array.
    vertex_graph = {0: [1, 2, 4],
                    1: [0, 3, 5],
                    2: [0, 3, 6],
                    3: [1, 2, 7],
                    4: [0, 5, 6],
                    5: [1, 4, 7],
                    6: [2, 4, 7]}
    # Note that this can be sped up by not drawing duplicate lines
    for vertex_idx in vertex_graph:
        neighbour_index = vertex_graph[vertex_idx]
        from_pos2d = vertices_2d[vertex_idx]
        for neighbour_idx in neighbour_index:
            to_pos2d = vertices_2d[neighbour_idx]
            if from_pos2d is None or to_pos2d is None:
                continue
            y1, x1 = from_pos2d[0], from_pos2d[1]
            y2, x2 = to_pos2d[0], to_pos2d[1]
            # Only stop drawing lines if both are outside
            if not point_in_canvas((y1, x1)) and not point_in_canvas((y2, x2)):
                continue

            for x, y in get_line(x1, y1, x2, y2):
                if point_in_canvas((y, x)):
                    image[int(y), int(x)] = (0, 0, 255)

            if point_in_canvas((y1, x1)):
                image[int(y1), int(x1)] = (255, 0, 255)
            if point_in_canvas((y2, x2)):
                image[int(y2), int(x2)] = (255, 0, 255)


def get_line(x1, y1, x2, y2):
    """
        根据两个平面点坐标生成两点之间直线上的点集

            参数：
                x1：点1在x方向上的坐标
                y1：点1在y方向上的坐标
                x2：点2在x方向上的坐标
                y2：点2在y方向上的坐标

            返回：
                points：两点之间直线上的点集

    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # print("Calculating line from {},{} to {},{}".format(x1,y1,x2,y2))
    points = []
    is_steep = abs(y2 - y1) > abs(x2 - x1)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    delta_x = x2 - x1
    delta_y = abs(y2 - y1)
    error = int(delta_x / 2)
    y = y1
    if y1 < y2:
        y_step = 1
    else:
        y_step = -1
    for x in range(x1, x2 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= delta_y
        if error < 0:
            y += y_step
            error += delta_x
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def set_point_in_canvas(point):
    """
        将超出图片范围的点设置到图片内离该点最近的位置中

            参数：
                point：像素坐标系下点的坐标

            返回：
                x：图片中离输入点最近处的点x坐标
                y：图片中离输入点最近处的点y坐标
    """
    x, y = point[0], point[1]

    if x < 0:
        x = 0
    elif x >= WINDOW_WIDTH:
        x = WINDOW_WIDTH - 1

    if y < 0:
        y = 0
    elif y >= WINDOW_HEIGHT:
        y = WINDOW_HEIGHT - 1

    return x, y


def point_in_canvas(point):
    """
        判断输入点是否在图片内

            参数：
                point：像素坐标系下点的坐标

            返回：
                bool：若在图片内，则返回True;反之输出False
    """
    if (point[0] >= 0) and (point[0] < WINDOW_HEIGHT) and (point[1] >= 0) and (point[1] < WINDOW_WIDTH):
        return True
    return False


def draw_3d_bounding_box_on_point_cloud(point_cloud, vertices_2d):
    """
        在点云上绘制3D边界框

            参数：
                point_cloud：点云数据，包含(x, y, z)坐标
                vertices_2d：3D边界框的8个顶点的像素坐标
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建边界框的线段
    lines = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            if (i, j) in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
                          (4, 5), (4, 6), (5, 7), (6, 7)]:
                lines.append([vertices_2d[i], vertices_2d[j]])

    # 创建线段对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices_2d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 设置线段颜色为红色

    # 可视化点云和边界框
    o3d.visualization.draw_geometries([pcd, line_set])

def plot_segmentation_results(seg_path, img_path):
    """主函数：绘制分割结果"""
    # 读取图像
    img_array = np.array(Image.open(seg_path))
    
    # 创建各分类mask
    sky_mask = create_mask(img_array, SKY_IDS)
    rigid_mask = create_mask(img_array, RIGID_IDS, exclude_ego=True)
    nonrigid_mask = create_mask(img_array, NONRIGID_IDS)
    dynamic_mask = rigid_mask | nonrigid_mask
    ego_mask = get_ego_mask(img_array[:, :, 0])
    
    # 创建绘图布局
    plt.figure(figsize=(15, 8))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(Image.open(img_path))
    plt.title('Original Image')
    plt.axis('off')
    
    # 动态物体
    plt.subplot(2, 3, 2)
    plt.imshow(dynamic_mask, cmap='gray')
    plt.title('Dynamic')
    plt.axis('off')
    
    # 天空区域
    plt.subplot(2, 3, 3)
    plt.imshow(sky_mask, cmap='gray')
    plt.title('Sky')
    plt.axis('off')
    
    # 刚性动态物体
    plt.subplot(2, 3, 4)
    plt.imshow(rigid_mask, cmap='gray')
    plt.title('Rigid Dynamic')
    plt.axis('off')
    
    # 非刚性动态物体
    plt.subplot(2, 3, 5)
    plt.imshow(nonrigid_mask, cmap='gray')
    plt.title('Nonrigid')
    plt.axis('off')
    
    # Ego Mask
    plt.subplot(2, 3, 6)
    plt.imshow(ego_mask, cmap='gray')
    plt.title('Ego Vehicle Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_mask(img_array, ids, exclude_ego=False):
    """创建指定类别的mask"""
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    for id in ids:
        mask |= img_array[:, :, 0] == id
    
    if exclude_ego:
        ego_mask = get_ego_mask(img_array[:, :, 0])
        mask[ego_mask] = False
    
    return mask

def get_ego_mask(img_array):
    """获取ego车辆mask"""
    working_array = img_array.astype(np.uint8) * 255
    height, width = working_array.shape[:2]
    seed_point = (height - 1, width // 2)
    ego_mask = flood_fill(working_array, seed_point, 0, tolerance=10)
    return ego_mask == 0

def view_images(image_dir, image_ext='_camera_0.png', window_name='Image Sequence', delay=200):
    """
    查看指定目录下的图片序列
    
    参数:
        image_dir (str): 图片目录路径
        image_ext (str): 图片扩展名，默认为'.png'
        window_name (str): 显示窗口的名称，默认为'Image Sequence'
        delay (int): 图片显示时间间隔（毫秒），默认为200ms
    """
    # 获取所有指定扩展名的图片
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(image_ext)])
    
    # 遍历并显示图片
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
            
        cv2.imshow(window_name, img)
        
        # 按任意键继续，按'q'退出
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def depth_to_pointcloud(depth_path, camera_matrix, show=True):
    """
    将深度图转换为点云并可视化
    参数：
        depth_path: 深度图路径（16位PNG）
        camera_matrix: 3x4相机投影矩阵（支持P0-P3）
        show: 是否显示点云
    """
    max_distance = 250
    # 解析相机参数矩阵
    K = camera_matrix.reshape(3,4)[:3,:3]  # 提取内参矩阵
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 读取深度图（16位PNG，单位毫米）
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_image.astype(np.float32)    # 转换为米
    print(depth)

    # 矢量化计算提升性能
    u = np.arange(depth.shape[1])
    v = np.arange(depth.shape[0])
    u, v = np.meshgrid(u, v)
    
    # 计算三维坐标
    z = depth
    valid_mask = z > 0
    if max_distance is not None:
        valid_mask = valid_mask & (depth <= max_distance)

    x = (u[valid_mask] - cx) * z[valid_mask] / fx
    y = (v[valid_mask] - cy) * z[valid_mask] / fy
    points = np.column_stack((x, y, z[valid_mask]))

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if show:
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f'Point Cloud: {os.path.basename(depth_path)}')
        
        # 添加坐标系和几何体
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # 设置视角参数
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, -1, 0])  # 相机朝向正前方
        view_ctl.set_up([0, 0, 1])      # 上方向为Z轴
        view_ctl.set_zoom(0.5)          # 初始缩放比例
        
        # 运行可视化
        vis.run()
        vis.destroy_window()

    return pcd

def read_velodyne_bin(bin_path):
    """ 读取KITTI格式的雷达bin文件 """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # 取x,y,z坐标

def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    points_2d = []
    for point in points_3d:
        X, Y, Z = point
        x = (fx * X / Z) + cx
        y = (fy * Y / Z) + cy
        points_2d.append((x, y))
    return np.array(points_2d)


def project_lidar_to_camera(bin_path, img_path, cam_intrinsic, Tr_velo_to_cam, max_depth=100.0):
    """
    将雷达点云投影到相机图像并生成mask和深度图
    
    参数：
        bin_path: 雷达点云路径(.bin)
        img_path: 对应相机图像路径
        cam_intrinsic: 相机内参矩阵(3x3)
        Tr_velo_to_cam: 雷达到相机的外参矩阵(3x4)
        max_depth: 最大有效深度（米）
        
    返回：
        mask: 投影点存在的像素位置（布尔矩阵）
        depth_map: 对应位置的深度值（单位：米）
        overlap_img: 投影可视化图像
    """
    # 读取数据
    points_velo = read_velodyne_bin(bin_path)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # 转换外参矩阵为齐次坐标形式 (4x4)
    Tr = np.vstack([Tr_velo_to_cam.reshape(3,4), [0, 0, 0, 1]])

    # 坐标系转换：雷达坐标系 → 相机坐标系
    ones = np.ones((points_velo.shape[0], 1))
    points_velo_hom = np.hstack([points_velo, ones])
    points_cam = (Tr @ points_velo_hom.T).T[:, :3]

    # 过滤有效点（z>0且在最大深度内）
    valid = (points_cam[:, 2] > 0) & (points_cam[:, 2] <= max_depth)
    points_cam = points_cam[valid]

    # 使用优化后的投影函数
    fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
    cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]
    uv = project_3d_to_2d(points_cam, fx, fy, cx, cy)
    uv = np.floor(uv).astype(int)

    # 过滤图像范围内的点
    in_view = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[in_view]
    depths = points_cam[in_view, 2]

    # 初始化输出矩阵
    mask = np.zeros((height, width), dtype=bool)
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # 可视化图像（原始图像叠加投影点）
    overlap_img = img.copy()
    
    # 更新mask和depth（保留最近点的深度）
    for (u, v), d in zip(uv, depths):
        if depth_map[v, u] == 0 or d < depth_map[v, u]:
            depth_map[v, u] = d
            mask[v, u] = True
            # 用颜色表示深度（红色越深表示越近）
            color = (0, 0, 255 * (1 - d/max_depth))
            cv2.circle(overlap_img, (u, v), 1, color, -1)

    return mask, depth_map, overlap_img

def save_depth_and_mask(depth_map, mask, overlap_img, data_root, frame_id, camera_id):
    """
    保存深度图、mask和overlap_img到data_root/lidar_depth目录
    
    参数：
        depth_map: 深度图矩阵
        mask: 掩码矩阵
        overlap_img: 投影可视化图像
        data_root: 数据根目录路径
        frame_id: 帧ID
        camera_id: 相机ID
    """
    import os
    import numpy as np
    import cv2
    
    # 创建保存目录
    save_dir = os.path.join(data_root, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)
    
    # 将mask和value保存为字典
    lidar_data = {
        'mask': mask,
        'value': depth_map[mask]  # 只保存有效点的深度值
    }
    
    # 保存为npy文件
    npy_path = os.path.join(save_dir, f'{frame_id:06}_camera_{camera_id}.npy')
    np.save(npy_path, lidar_data)
    
    # 保存overlap_img
    overlap_path = os.path.join(save_dir, f'{frame_id:06}_camera_{camera_id}_overlap.png')
    cv2.imwrite(overlap_path, overlap_img)
    
    print(f"数据已保存到: {npy_path} 和 {overlap_path}")

def depth_to_pcd(data_root, frame_id, camera_id):
    """
    从保存的npy文件生成点云并可视化
    
    参数：
        data_root: 数据根目录路径
        frame_id: 帧ID
        camera_id: 相机ID
    """
    import numpy as np
    import cv2
    
    # 加载保存的数据
    save_dir = os.path.join(data_root, 'lidar_depth')
    npy_path = os.path.join(save_dir, f'{frame_id:06}_camera_{camera_id}.npy')
    lidar_data = np.load(npy_path, allow_pickle=True).item()
    
    # 解析内参参数（使用默认值，可根据实际情况修改）
    fx = 960.0
    fy = 960.0
    cx = 960.0
    cy = 540.0

    # 获取有效像素坐标
    valid_coords = np.argwhere(lidar_data['mask'])
    u = valid_coords[:, 1]  # 宽度方向 (列索引)
    v = valid_coords[:, 0]  # 高度方向 (行索引)
    Z = lidar_data['value'] # 深度值 (单位：米)

    # 核心转换公式
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    points = np.column_stack((X, Y, Z))

    # Open3D可视化
    try:
        import open3d as o3d
        # 生成有效点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) 

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='LiDAR Depth Projection', width=1280, height=720)
        
        # 添加坐标系（缩小尺寸避免遮挡）
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(coord_frame)
        vis.add_geometry(pcd)

        # 设置优化后的视角参数
        # view_ctl = vis.get_view_control()
        # view_ctl.set_front([-0.5, -0.3, 0.8])  # 最佳观测角度
        # view_ctl.set_up([0, 0, 1])             # 保持Z轴向上
        # view_ctl.set_zoom(0.3)                 # 适合车辆尺寸的缩放

        # 添加点云着色
        pcd.paint_uniform_color([0.2, 0.7, 0.3])  # 绿色点云
        
        # 运行可视化
        print("\n按Q退出可视化窗口...")
        vis.run()
        vis.destroy_window()

        # 输出统计信息
        print("\n=== 点云统计 ===")
        print(f"有效点数: {len(points)}")
        print(f"X范围: [{np.min(X):.2f}, {np.max(X):.2f}] m")
        print(f"Y范围: [{np.min(Y):.2f}, {np.max(Y):.2f}] m")
        print(f"Z范围: [{np.min(Z):.2f}, {np.max(Z):.2f}] m")

    except ImportError:
        print("请安装open3d: pip install open3d")
    except Exception as e:
        print(f"可视化错误: {str(e)}")

    return points

import cv2
import os

def images_to_video(data_root, start_frame, end_frame, frame_rate=30):
    """
    将图片序列转换为视频

    参数:
        data_root (str): 数据根目录路径
        start_frame (int): 起始帧ID
        end_frame (int): 结束帧ID
        frame_rate (int): 视频帧率，默认为30
    """
    # 创建视频保存目录
    video_dir = os.path.join(data_root, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    # 视频输出路径
    video_path = os.path.join(video_dir, 'output_video.mp4')
    
    # 获取图片路径列表
    img_paths = [f"{data_root}/image/{frame_id:06}_camera_0.png" for frame_id in range(start_frame, end_frame + 1)]
    
    # 读取第一张图片以获取视频尺寸
    first_img = cv2.imread(img_paths[0])
    print(f"图片尺寸: {first_img.shape}")
    height, width, layers = first_img.shape
    
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    
    total_frames = len(img_paths)
    progress_bar = tqdm(total=total_frames, desc="生成视频进度", unit="帧")

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            progress_bar.update(1)  # 即使读取失败也更新进度
            continue
        video_writer.write(img)
        progress_bar.update(1)  # 更新进度条
        progress_bar.set_postfix({"当前帧": os.path.basename(img_path)})  # 显示正在处理的文件名

    progress_bar.close()  # 关闭进度条
    
    # 释放视频写入对象
    video_writer.release()
    print(f"视频已保存到: {video_path}")

def new_view_to_video(data_root, start_frame, end_frame, frame_rate=30):
    """
    将图片序列转换为视频

    参数:
        data_root (str): 数据根目录路径
        start_frame (int): 起始帧ID
        end_frame (int): 结束帧ID
        frame_rate (int): 视频帧率，默认为30
    """
    # 创建视频保存目录
    video_dir = os.path.join(data_root, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    # 视频输出路径
    video_path = os.path.join(video_dir, 'new_view_3.mp4')
    
    # 获取图片路径列表
    img_paths = [f"{data_root}/new_view/{frame_id:06}_new_view_3.png" for frame_id in range(start_frame, end_frame + 1)]
    
    # 读取第一张图片以获取视频尺寸
    first_img = cv2.imread(img_paths[0])
    print(f"图片尺寸: {first_img.shape}")
    height, width, layers = first_img.shape
    
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    
    total_frames = len(img_paths)
    progress_bar = tqdm(total=total_frames, desc="生成视频进度", unit="帧")

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            progress_bar.update(1)  # 即使读取失败也更新进度
            continue
        video_writer.write(img)
        progress_bar.update(1)  # 更新进度条
        progress_bar.set_postfix({"当前帧": os.path.basename(img_path)})  # 显示正在处理的文件名

    progress_bar.close()  # 关闭进度条
    
    # 释放视频写入对象
    video_writer.release()
    print(f"视频已保存到: {video_path}")

def generate_new_view_videos(data_root, start_frame, end_frame, frame_rate=30):
    """
    批量生成多视角视频
    
    参数:
        data_root (str): 数据根目录路径
        start_frame (int): 起始帧ID
        end_frame (int): 结束帧ID
        frame_rate (int): 视频帧率，默认为30
    """
    # 并行处理所有视角（0-9）
    for view_id in range(10):  # 0到9共10个视角
        generate_single_view_video(data_root, start_frame, end_frame, view_id, frame_rate)

def generate_single_view_video(data_root, start_frame, end_frame, view_id, frame_rate=30):
    """
    生成单个视角的视频
    
    参数:
        view_id (int): 视角编号（0-9）
    """
    # 创建视频保存目录
    video_dir = os.path.join(data_root, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    # 视频输出路径
    video_path = os.path.join(video_dir, f'new_view_{view_id+1}.mp4')
    
    # 获取图片路径列表
    img_paths = [
        f"{data_root}/new_view/{frame_id:06}_new_view_{view_id}.png" 
        for frame_id in range(start_frame, end_frame + 1)
    ]
    
    # 读取第一张图片获取视频尺寸
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        print(f"无法读取视角{view_id}的首帧图片: {img_paths[0]}")
        return
    
    height, width = first_img.shape[:2]
    
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    
    # 进度条显示视角信息
    total_frames = len(img_paths)
    progress_bar = tqdm(
        total=total_frames, 
        desc=f"生成视角{view_id}视频进度",
        unit="帧"
    )

    # 写入视频帧
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"缺失帧: {os.path.basename(img_path)}")
            progress_bar.update(1)
            continue
            
        video_writer.write(img)
        progress_bar.update(1)
        progress_bar.set_postfix({
            "当前帧": os.path.basename(img_path),
            "分辨率": f"{width}x{height}"
        })

    # 释放资源
    progress_bar.close()
    video_writer.release()
    print(f"视角{view_id}视频已保存到: {video_path}")

def read_calibration(file_path):
    """读取标定文件中的所有相机和雷达数据"""
    calibration_data = {
        'P0': None,
        'P1': None,
        'P2': None,
        'P3': None,
        'R0_rect': None,
        'Tr_velo_to_cam': {}
    }

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                calibration_data['P0'] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
            elif line.startswith('P1:'):
                calibration_data['P1'] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
            elif line.startswith('P2:'):
                calibration_data['P2'] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
            elif line.startswith('P3:'):
                calibration_data['P3'] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
            elif line.startswith('Tr_velo_to_cam_'):
                # 提取相机编号
                cam_id = int(line.split(':')[0].split('_')[-1])
                calibration_data['Tr_velo_to_cam'][cam_id] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
            elif line.startswith('TR_imu_to_velo:'):
                calibration_data['TR_imu_to_velo'] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)

    return calibration_data

def get_lidar_to_camera_transform(data_root, camera_id):
    """
    计算雷达到相机的转换矩阵
    
    参数：
        data_root: 数据根目录路径
        camera_id: 相机ID
        
    返回：
        Tr_velo_to_cam: 3x4 雷达到相机的转换矩阵
    """
    # 从extrinsic文件夹读取相机外参矩阵
    camera_extrinsic_path = f"{data_root}/extrinsic/{camera_id:03}.txt"
    camera_extrinsic = np.loadtxt(camera_extrinsic_path)

    # 从extrinsic文件夹读取雷达外参矩阵
    lidar_extrinsic_path = f"{data_root}/extrinsic/005.txt"
    lidar_extrinsic = np.loadtxt(lidar_extrinsic_path)

    # 计算雷达外参矩阵的逆
    lidar_extrinsic_inv = np.linalg.inv(lidar_extrinsic)

    # 计算雷达到相机的CARLA坐标系转换矩阵
    lidar_to_camera_carla = np.dot(camera_extrinsic, lidar_extrinsic_inv)

    # CARLA 到 KITTI 坐标系转换矩阵
    # CARLA (x,y,z) -> KITTI (z,x,-y)
    velo_to_cam = np.array([
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])

    # 最终的雷达到相机（KITTI坐标系）的转换矩阵
    Tr_velo_to_cam_full = np.dot(velo_to_cam, lidar_to_camera_carla)

    # 提取所需的3x4矩阵
    Tr_velo_to_cam = Tr_velo_to_cam_full[:3, :]
    print(Tr_velo_to_cam)

    return Tr_velo_to_cam

def colorize_lidar_pcd(mask, depth_map, img_array, cam_intrinsic, max_depth=100.0, show=True):
    """
    使用投影结果生成彩色点云
    
    参数：
        mask: 有效点掩码矩阵 (H,W)
        depth_map: 深度图矩阵 (H,W)
        img_array: RGB图像数组 (H,W,3)
        cam_intrinsic: 相机内参矩阵 (3x3)
        max_depth: 最大有效深度（米）
        show: 是否显示点云
    """
    # 解析相机参数
    fx, fy = cam_intrinsic[0,0], cam_intrinsic[1,1]
    cx, cy = cam_intrinsic[0,2], cam_intrinsic[1,2]

    # 获取有效点坐标
    valid_coords = np.argwhere(mask)
    u = valid_coords[:, 1]  # 列坐标 (x)
    v = valid_coords[:, 0]  # 行坐标 (y)
    Z = depth_map[mask]     # 深度值

    # 过滤有效深度
    valid_depth = Z <= max_depth
    u = u[valid_depth]
    v = v[valid_depth]
    Z = Z[valid_depth]

    # 计算3D坐标
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.column_stack((X, Y, Z))

    # 获取颜色信息
    colors = img_array[v, u] / 255.0  # 归一化到0-1

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if show:
        # 可视化设置
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Colored LiDAR Point Cloud', width=1280, height=720)
        
        # 添加坐标系（缩小尺寸）
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(coord_frame)
        vis.add_geometry(pcd)
        # vis.get_render_option().point_size = 0.5

        # 设置视角参数
        view_ctl = vis.get_view_control()
        view_ctl.set_front([-0.5, -0.3, 0.8])  # 最佳观测角度
        view_ctl.set_up([0, 1, 0])             # y轴向上
        view_ctl.set_zoom(0.3)

        # 运行可视化
        print("按Q退出可视化窗口...")
        vis.run()
        vis.destroy_window()

    return pcd

"""
CARLA Waypoint可视化工具类
包含所有与waypoint显示相关的功能
"""

class WaypointVisualizer:
    """
    Waypoint可视化工具类
    """
    
    def __init__(self, world, debug=None, waypoint_lifetime=60.0):
        """
        初始化可视化工具
        
        参数:
            world: CARLA世界对象
            debug: CARLA debug接口，如果为None则自动获取
            waypoint_lifetime: waypoint显示时间（秒）
        """
        self.world = world
        self.debug = debug if debug is not None else world.debug
        self.waypoint_lifetime = waypoint_lifetime
        self.vehicle_path_history = []  # 车辆轨迹历史
        
    def draw_waypoints_with_numbers(self, waypoints, color=None, prefix=""):
        """
        绘制waypoints并显示编号（支持前缀）
        
        参数:
            waypoints: waypoint列表
            color: 颜色，如果为None则使用默认颜色
            prefix: 编号前缀
        """
        if self.debug is None:
            return
            
        if color is None:
            color = carla.Color(255, 0, 0)  # 红色
            
        for i, waypoint in enumerate(waypoints):
            location = waypoint.transform.location
            
            # 绘制waypoint点
            self.debug.draw_point(
                location + carla.Location(z=0.5),  # 稍微抬高一点
                size=0.1,
                color=color,
                life_time=self.waypoint_lifetime
            )
            
            # 绘制编号（每5个点显示一个，避免太密集）
            if i % 5 == 0:
                label = f"{prefix}{i}" if prefix else str(i)
                self.debug.draw_string(
                    location + carla.Location(z=1.5),  # 在点上方显示文字
                    label,
                    draw_shadow=True,
                    color=carla.Color(255, 255, 255),  # 白色文字
                    life_time=self.waypoint_lifetime
                )
            
            # 绘制方向箭头（每3个点一个箭头）
            if i < len(waypoints) - 1 and i % 3 == 0:
                next_location = waypoints[i + 1].transform.location
                self.debug.draw_arrow(
                    location + carla.Location(z=0.3),
                    next_location + carla.Location(z=0.3),
                    thickness=0.05,
                    arrow_size=0.2,
                    color=color,
                    life_time=self.waypoint_lifetime
                )
                
        
    def visualize_waypoints_around_vehicle(self, vehicle, distance=50.0, interval=2.0):
        """
        可视化车辆周围的waypoints并返回轨迹
        
        参数:
            vehicle: 车辆对象
            distance: 可视化范围（米）
            interval: waypoint间隔（米）
            
        返回:
            List[carla.Transform]: waypoint轨迹列表
        """
        if vehicle is None or self.debug is None:
            return []
            
        # 获取车辆当前位置的waypoint
        current_location = vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(current_location)
        
        if current_waypoint is None:
            return []
            
        # 生成前方waypoints
        waypoints = [current_waypoint]
        wp = current_waypoint
        
        # 向前生成waypoints
        forward_distance = 0
        while forward_distance < distance:
            next_wps = wp.next(interval)
            if next_wps:
                wp = next_wps[0]
                waypoints.append(wp)
                forward_distance += interval
            else:
                break
                
        # 向后生成waypoints
        wp = current_waypoint
        backward_distance = 0
        while backward_distance < distance / 2:  # 向后显示一半距离
            prev_wps = wp.previous(interval)
            if prev_wps:
                wp = prev_wps[0]
                waypoints.insert(0, wp)  # 插入到开头
                backward_distance += interval
            else:
                break
        
        # 可视化waypoints（保持原有功能）
        self.draw_waypoints_with_numbers(waypoints, carla.Color(255, 0, 0), "C")
        
        # 返回Transform列表
        return [wp.transform for wp in waypoints]

    def visualize_lane_change_path(self, start_waypoint, target_lateral_offset, num_steps=10):
        """
        可视化变道路径并返回轨迹
        
        参数:
            start_waypoint: 起始waypoint
            target_lateral_offset: 目标横向偏移（正值向右，负值向左）
            num_steps: 变道步骤数
            
        返回:
            dict: 包含三段轨迹的字典
                - 'original': 原始车道轨迹 List[carla.Transform]
                - 'lane_change': 变道轨迹 List[carla.Transform]  
                - 'after_change': 变道后轨迹 List[carla.Transform]
        """
        if self.debug is None:
            return {'original': [], 'lane_change': [], 'after_change': []}
            
        result_trajectories = {'original': [], 'lane_change': [], 'after_change': []}
        
        # 生成原始车道轨迹
        current_wp = start_waypoint
        original_waypoints = [current_wp]
        traveled_distance = 0.0
        target_distance = 30.0  # 10米
        
        while traveled_distance < target_distance:
            next_wps = current_wp.next(2.0)
            if next_wps:
                current_wp = next_wps[0]
                original_waypoints.append(current_wp)
                traveled_distance += 2.0
            else:
                break
        
        # 绘制前10米的原始路径（蓝色）
        self.draw_waypoints_with_numbers(original_waypoints, carla.Color(0, 100, 255), "Orig")
        result_trajectories['original'] = [wp.transform for wp in original_waypoints]
        
        # 从10米外的waypoint开始生成变道路径
        lane_change_start_wp = current_wp
        
        # 生成变道路径
        lane_change_transforms = []
        for i in range(num_steps):
            # 计算当前的横向偏移
            progress = i / (num_steps - 1) if num_steps > 1 else 0
            current_offset = target_lateral_offset * self._smooth_step(progress)
            
            # 计算偏移后的位置
            right_vector = current_wp.transform.get_right_vector()
            offset_location = current_wp.transform.location + carla.Location(
                x=current_offset * right_vector.x,
                y=current_offset * right_vector.y,
                z=0.5
            )
            
            # 创建Transform对象
            lane_change_transform = carla.Transform(offset_location, current_wp.transform.rotation)
            lane_change_transforms.append(lane_change_transform)
            
            # 创建虚拟waypoint用于可视化
            class VirtualWaypoint:
                def __init__(self, location):
                    self.transform = carla.Transform(location)
                    
            virtual_wp = VirtualWaypoint(offset_location)
            
            # 移动到下一个waypoint
            next_wps = current_wp.next(2.0)
            if next_wps:
                current_wp = next_wps[0]
            else:
                break
        
        # 使用橙色绘制变道路径
        waypoints_for_viz = []
        for transform in lane_change_transforms:
            class VirtualWaypoint:
                def __init__(self, transform):
                    self.transform = transform
            waypoints_for_viz.append(VirtualWaypoint(transform))
        
        self.draw_waypoints_with_numbers(waypoints_for_viz, carla.Color(255, 165, 0), "Change")
        result_trajectories['lane_change'] = lane_change_transforms
        
        # 变道完成后，继续在新车道行驶10米
        if lane_change_transforms:
            # 从变道结束位置开始，找到对应的道路waypoint
            last_change_location = lane_change_transforms[-1].location
            
            # 在新车道中找到最近的waypoint
            new_lane_wp = self.world.get_map().get_waypoint(last_change_location)
            
            if new_lane_wp:
                after_change_waypoints = [new_lane_wp]
                traveled_distance = 0.0
                target_distance = 50.0  # 变道后再行驶10米
                
                while traveled_distance < target_distance:
                    next_wps = new_lane_wp.previous(2.0)
                    if next_wps:
                        new_lane_wp = next_wps[0]
                        after_change_waypoints.append(new_lane_wp)
                        traveled_distance += 2.0
                    else:
                        break
                
                # 绘制变道后的路径（绿色）
                self.draw_waypoints_with_numbers(after_change_waypoints, carla.Color(0, 255, 100), "After")
                result_trajectories['after_change'] = [wp.transform for wp in after_change_waypoints]
        
        return result_trajectories
    

    def _smooth_step(self, t):
        """
        平滑步进函数，创建S形曲线
        """
        return 0.5 * (1 + math.sin(math.pi * (t - 0.5)))

    def visualize_vehicle_path_history(self, vehicle, max_points=50):
        """
        可视化车辆行驶轨迹
        
        参数:
            vehicle: 车辆对象
            max_points: 最大轨迹点数
        """
        if vehicle is None or self.debug is None:
            return
            
        # 添加当前位置到轨迹
        current_location = vehicle.get_location()
        self.vehicle_path_history.append(current_location)
        
        # 限制轨迹长度
        if len(self.vehicle_path_history) > max_points:
            self.vehicle_path_history.pop(0)
            
        # 绘制轨迹
        for i, location in enumerate(self.vehicle_path_history):
            # 颜色从暗到亮，表示时间先后
            intensity = int(255 * (i + 1) / len(self.vehicle_path_history))
            color = carla.Color(intensity, intensity, 0)  # 黄色渐变
            
            self.debug.draw_point(
                location + carla.Location(z=0.2),
                size=0.05,
                color=color,
                life_time=1.0
            )
            
    def show_nearby_lanes(self, vehicle, distance=50.0, interval=2.0):
        """
        显示附近车道waypoints（修复对向车道显示问题）
        
        参数:
            vehicle: 车辆对象
            distance: 显示距离
            interval: waypoint间隔
        """
        if vehicle is None or self.debug is None:
            return
            
        # 获取当前waypoint
        current_location = vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_location)
        
        if current_wp is None:
            return
        
        print(f"当前车道ID: {current_wp.lane_id}")
        
        # 显示左侧车道
        left_wp = current_wp.get_left_lane()
        if left_wp and left_wp.lane_type == carla.LaneType.Driving:
            is_opposite = self._check_opposite(current_wp, left_wp)
            left_waypoints = self._gen_waypoints(left_wp, vehicle, distance, interval, is_opposite)
            print(f"左侧车道waypoints数量: {len(left_waypoints)}")
            color = carla.Color(255, 150, 150) if is_opposite else carla.Color(0, 255, 0)
            prefix = "OL" if is_opposite else "L"
            self.draw_waypoints_with_numbers(left_waypoints, color, prefix)
            print(f"左侧车道ID: {left_wp.lane_id} ({'对向' if is_opposite else '同向'})")
        
        # 显示右侧车道
        right_wp = current_wp.get_right_lane()
        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
            is_opposite = self._check_opposite(current_wp, right_wp)
            right_waypoints = self._gen_waypoints(right_wp, vehicle, distance, interval, is_opposite)
            print(f"右侧车道waypoints数量: {len(right_waypoints)}")
            color = carla.Color(150, 150, 255) if is_opposite else carla.Color(0, 0, 255)
            prefix = "OR" if is_opposite else "R"
            self.draw_waypoints_with_numbers(right_waypoints, color, prefix)
            print(f"右侧车道ID: {right_wp.lane_id} ({'对向' if is_opposite else '同向'})")

    def _check_opposite(self, current_wp, target_wp):
        """
        改进的对向车道检测方法
        
        参数:
            current_wp: 当前waypoint
            target_wp: 目标waypoint
            
        返回:
            bool: 是否为对向车道
        """
        # 方法1：通过朝向角度判断
        current_yaw = current_wp.transform.rotation.yaw
        target_yaw = target_wp.transform.rotation.yaw
        
        # 计算角度差
        angle_diff = abs(current_yaw - target_yaw)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 方法2：通过lane_id符号判断（CARLA中通常正负表示不同方向）
        lane_id_opposite = (current_wp.lane_id > 0) != (target_wp.lane_id > 0)
        
        # 综合判断
        angle_opposite = 160 <= angle_diff <= 200
        is_opposite = angle_opposite or lane_id_opposite
        
        if is_opposite:
            print(f"  检测到对向车道: 当前朝向={current_yaw:.1f}°, 目标朝向={target_yaw:.1f}°, 角度差={angle_diff:.1f}°")
            print(f"  车道ID: 当前={current_wp.lane_id}, 目标={target_wp.lane_id}, ID符号相反={lane_id_opposite}")
        
        return is_opposite

    def _gen_waypoints(self, start_wp, vehicle, distance, interval, is_opposite=False):
        """
        改进的waypoint生成方法，修复对向车道显示问题
        
        参数:
            start_wp: 起始waypoint
            vehicle: 车辆对象（用于获取ego位置）
            distance: 生成距离
            interval: waypoint间隔
            is_opposite: 是否为对向车道
            
        返回:
            list: waypoint列表
        """
        waypoints = []
        
        if is_opposite:
            # 对向车道：需要找到在ego车辆前方但朝向相反的路径点
            ego_location = vehicle.get_location()
            
            # 找到对向车道中距离ego车辆最近的点作为起始点
            min_distance = float('inf')
            best_wp = start_wp
            
            # 在对向车道上搜索与ego位置最接近的waypoint
            search_wp = start_wp
            search_attempts = 0
            max_search_attempts = int(distance * 3 / interval)  # 扩大搜索范围
            
            # 先向前搜索
            while search_attempts < max_search_attempts:
                dist = search_wp.transform.location.distance(ego_location)
                if dist < min_distance:
                    min_distance = dist
                    best_wp = search_wp
                
                next_wps = search_wp.next(interval)
                if next_wps:
                    search_wp = next_wps[0]
                    search_attempts += 1
                else:
                    break
            
            # 再向后搜索
            search_wp = start_wp
            search_attempts = 0
            while search_attempts < max_search_attempts:
                dist = search_wp.transform.location.distance(ego_location)
                if dist < min_distance:
                    min_distance = dist
                    best_wp = search_wp
                
                prev_wps = search_wp.previous(interval)
                if prev_wps:
                    search_wp = prev_wps[0]
                    search_attempts += 1
                else:
                    break
            
            print(f"  对向车道最近点距离ego: {min_distance:.2f}m")
            
            # 从最佳起始点开始生成路径点
            waypoints = [best_wp]
            wp = best_wp
            traveled = 0
            
            # 向ego车辆前进方向生成waypoints（对向车道的前进方向）
            while traveled < distance and len(waypoints) < 50:
                next_wps = wp.next(interval)
                if next_wps:
                    wp = next_wps[0]
                    waypoints.append(wp)
                    traveled += interval
                else:
                    break
                    
        else:
            # 同向车道：正常的前向生成
            waypoints = [start_wp]
            wp = start_wp
            traveled = 0
            
            while traveled < distance and len(waypoints) < 50:
                next_wps = wp.next(interval)
                if next_wps:
                    wp = next_wps[0]
                    waypoints.append(wp)
                    traveled += interval
                else:
                    break
        
        print(f"  生成了 {len(waypoints)} 个waypoints ({'对向' if is_opposite else '同向'})")
        return waypoints

    def show_all_lanes(self, vehicle, distance=50.0, max_lanes=2):
        """
        显示所有车道
        
        参数:
            vehicle: 车辆对象
            distance: 显示距离
            max_lanes: 最大车道数
        """
        if vehicle is None or self.debug is None:
            return
            
        current_location = vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_location)
        
        if current_wp is None:
            return
        
        colors = [
            carla.Color(255, 100, 100),  # 淡红
            carla.Color(100, 255, 100),  # 淡绿  
            carla.Color(100, 100, 255),  # 淡蓝
            carla.Color(255, 255, 100),  # 淡黄
        ]
        
        # 当前车道
        current_waypoints = self._gen_waypoints(current_wp, vehicle, distance, 2.0)
        self.draw_waypoints_with_numbers(current_waypoints, carla.Color(255, 0, 0), "C")
        
        # 左侧车道
        wp = current_wp
        for i in range(max_lanes):
            left_wp = wp.get_left_lane()
            if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                is_opposite = self._check_opposite(current_wp, left_wp)
                waypoints = self._gen_waypoints(left_wp, vehicle, distance, 3.0, is_opposite)
                
                color = carla.Color(255, 150, 150) if is_opposite else colors[i % len(colors)]
                prefix = f"OL{i+1}" if is_opposite else f"L{i+1}"
                
                self.draw_waypoints_with_numbers(waypoints, color, prefix)
                wp = left_wp
                print(f"左侧车道{i+1}: ID={left_wp.lane_id} ({'对向' if is_opposite else '同向'})")
            else:
                break
        
        # 右侧车道
        wp = current_wp
        for i in range(max_lanes):
            right_wp = wp.get_right_lane()
            if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                is_opposite = self._check_opposite(current_wp, right_wp)
                waypoints = self._gen_waypoints(right_wp, vehicle, distance, 3.0, is_opposite)
                
                color = carla.Color(150, 150, 255) if is_opposite else colors[(i + max_lanes) % len(colors)]
                prefix = f"OR{i+1}" if is_opposite else f"R{i+1}"
                
                self.draw_waypoints_with_numbers(waypoints, color, prefix)
                wp = right_wp
                print(f"右侧车道{i+1}: ID={right_wp.lane_id} ({'对向' if is_opposite else '同向'})")
            else:
                break

    def debug_lane_info(self, vehicle):
        """
        调试车道信息
        
        参数:
            vehicle: 车辆对象
        """
        if vehicle is None:
            return
            
        current_location = vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_location)
        
        if current_wp is None:
            return
        
        print("=== 车道调试信息 ===")
        print(f"当前位置: ({current_location.x:.2f}, {current_location.y:.2f})")
        print(f"当前车道ID: {current_wp.lane_id}")
        print(f"当前朝向: {current_wp.transform.rotation.yaw:.2f}°")
        
        # 检查左右车道
        left_wp = current_wp.get_left_lane()
        if left_wp:
            print(f"左侧车道ID: {left_wp.lane_id}, 朝向: {left_wp.transform.rotation.yaw:.2f}°")
            print(f"左侧车道类型: {left_wp.lane_type}")
            print(f"左侧位置: ({left_wp.transform.location.x:.2f}, {left_wp.transform.location.y:.2f})")
        
        right_wp = current_wp.get_right_lane()
        if right_wp:
            print(f"右侧车道ID: {right_wp.lane_id}, 朝向: {right_wp.transform.rotation.yaw:.2f}°")
            print(f"右侧车道类型: {right_wp.lane_type}")
            print(f"右侧位置: ({right_wp.transform.location.x:.2f}, {right_wp.transform.location.y:.2f})")
        
        print("==================")

    def test_lane_visualization(self, vehicle):
        """
        测试车道可视化
        
        参数:
            vehicle: 车辆对象
        """
        if vehicle is None or self.debug is None:
            print("Vehicle或debug接口未初始化")
            return
            
        current_location = vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_location)
        
        if current_wp is None:
            print("无法获取当前waypoint")
            return
        
        print("开始测试车道可视化...")
        
        # 强制显示当前车道
        current_waypoints = self._gen_waypoints(current_wp, vehicle, 30.0, 2.0, False)
        self.draw_waypoints_with_numbers(current_waypoints, carla.Color(255, 0, 0), "CURR")
        print(f"当前车道显示了 {len(current_waypoints)} 个点")
        
        # 测试左侧车道
        left_wp = current_wp.get_left_lane()
        if left_wp and left_wp.lane_type == carla.LaneType.Driving:
            # 按对向处理
            left_waypoints_opp = self._gen_waypoints(left_wp, vehicle, 30.0, 2.0, True)
            self.draw_waypoints_with_numbers(left_waypoints_opp, carla.Color(0, 255, 0), "LOPP")
            print(f"左侧车道(对向方式)显示了 {len(left_waypoints_opp)} 个点")
            
            # 按同向处理
            left_waypoints_same = self._gen_waypoints(left_wp, vehicle, 30.0, 2.0, False)
            self.draw_waypoints_with_numbers(left_waypoints_same, carla.Color(0, 255, 255), "LSAME")
            print(f"左侧车道(同向方式)显示了 {len(left_waypoints_same)} 个点")

    def clear_debug_shapes(self):
        """
        清除所有debug绘制的形状（CARLA中通过life_time自动清除）
        """
        pass
        
    def update_visualization(self, vehicle):
        """
        更新可视化（每帧调用）
        
        参数:
            vehicle: 车辆对象
        """
        if vehicle:
            # 可视化车辆周围的waypoints
            self.visualize_waypoints_around_vehicle(vehicle, distance=30.0, interval=3.0)
            
            # 可视化车辆轨迹
            self.visualize_vehicle_path_history(vehicle)
            
    def show_waypoints_ahead(self, vehicle, distance=50.0):
        """
        手动显示前方waypoints
        
        参数:
            vehicle: 车辆对象
            distance: 显示距离
        """
        if vehicle:
            self.visualize_waypoints_around_vehicle(vehicle, distance=distance)

    def show_lane_change_preview(self, vehicle, lateral_offset=-3.5):
        """
        预览变道路径
        
        参数:
            vehicle: 车辆对象
            lateral_offset: 横向偏移量（负值向左，正值向右）
        """
        if vehicle:
            current_location = vehicle.get_location()
            current_waypoint = self.world.get_map().get_waypoint(current_location)
            if current_waypoint:
                self.visualize_lane_change_path(current_waypoint, lateral_offset)

if __name__ == "__main__":
    # seg_path = './data/training_20250305_130741/image/000000_camera_seg_0.png'
    # img_path = './data/training_20250305_130741/image/000000_camera_0.png'
    # plot_segmentation_results(seg_path, img_path)
    
    # # 新增图片查看功能示例
    # image_dir = './data/training_20250226_102047/image'
    # view_images(image_dir)
    
    # P0 = np.array([960.0, 0.0, 960.0, 0.0,
    #                0.0, 960.0, 540.0, 0.0,
    #                0.0, 0.0, 1.0, 0.0])
    
    # # 生成并显示点云（以P0相机为例）
    # point_cloud = depth_to_pointcloud(
    #     depth_path="./data/training_20250306_110708/depth/000000_depth_0.png",
    #     camera_matrix=P0,
    #     show=True
    # )
    
    # 输入参数
    frame_id = 1
    root = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20'
    data_root = f"{root}/training_20250508_103023_DynamicObjectCrossing_1"
    camera_id = 1
    lidar_id = 0
    
    # 路径配置
    bin_path = f"{data_root}/velodyne/{frame_id:06}_lidar_{lidar_id}.bin"
    img_path = f"{data_root}/image/{frame_id:06}_camera_{camera_id}.png"
    calib_path = f"{data_root}/calib/{frame_id:06}.txt"

    start_frame = 0
    end_frame = 250
    frame_rate = 24


    # images_to_video(data_root, start_frame, end_frame, frame_rate)
    # new_view_to_video(data_root, start_frame, end_frame, frame_rate)
    # generate_new_view_videos(data_root, start_frame=0, end_frame=250, frame_rate=24)


    calib_data = read_calibration(calib_path)

    # 获取相机内参矩阵 (P矩阵去掉最后一列)
    cam_intrinsic = calib_data[f'P{0}'][:, :3]

    Tr_velo_to_cam = get_lidar_to_camera_transform(data_root, camera_id)
    Tr_velo_to_cam[0, 3] = 0
    Tr_velo_to_cam[2, 3] = -1   
    # 执行投影
    mask, depth_map, overlap_img = project_lidar_to_camera(
        bin_path, img_path, cam_intrinsic, Tr_velo_to_cam, max_depth=100.0
    )

    # # # 保存结果
    # # cv2.imwrite(f"mask_{frame_id:06}.png", mask.astype(np.uint8)*255)
    # # np.save(f"depth_{frame_id:06}.npy", depth_map)
    # # cv2.imwrite(f"projection_{frame_id:06}.jpg", overlap_img)

    # 可视化
    cv2.imshow('Projection Overlay', overlap_img)
    key = cv2.waitKey(0)  # 等待按键
    if key == ord('q'):  # 按下 'q' 键退出
        cv2.destroyAllWindows()
    
    # # 保存深度图和mask
    # # 保存深度图、mask和overlap_img
    # frame_id = 1
    # camera_id = 0
    # save_depth_and_mask(depth_map, mask, overlap_img, data_root, frame_id, camera_id)

    # # 读取并可视化
    # depth_to_pcd(data_root, frame_id, camera_id)

    # 读取原始图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # 生成彩色点云
    colored_pcd = colorize_lidar_pcd(
        mask=mask,
        depth_map=depth_map,
        img_array=img_rgb,
        cam_intrinsic=cam_intrinsic,
        max_depth=100.0,
        show=True
    )
