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


config = yaml_to_config("configs.yaml")
WINDOW_WIDTH = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood_fill

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
        view_ctl = vis.get_view_control()
        view_ctl.set_front([-0.5, -0.3, 0.8])  # 最佳观测角度
        view_ctl.set_up([0, 0, 1])             # 保持Z轴向上
        view_ctl.set_zoom(0.3)                 # 适合车辆尺寸的缩放

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
    frame_id = 20
    data_root = "./data/training_20250313_103515"
    
    # 路径配置
    bin_path = f"{data_root}/velodyne/{frame_id:06}_lidar_0.bin"
    img_path = f"{data_root}/image/{frame_id:06}_camera_0.png"
    
    # 相机参数 (P0矩阵)
    cam_intrinsic = np.array([
        [960.0, 0.0, 960.0],
        [0.0, 960.0, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 外参矩阵 (Tr_velo_to_cam)
    Tr_velo_to_cam = np.array([
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    ]).reshape(3, 4)
    
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
    
    # 保存深度图和mask
    # 保存深度图、mask和overlap_img
    # frame_id = 1
    # camera_id = 0
    # save_depth_and_mask(depth_map, mask, overlap_img, data_root, frame_id, camera_id)

    # # 读取并可视化
    # depth_to_pcd(data_root, frame_id, camera_id)