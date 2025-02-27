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

if __name__ == "__main__":
    seg_path = './data/training_20250226_102047/image/000010_camera_seg_0.png'
    img_path = './data/training_20250226_102047/image/000010_camera_0.png'
    plot_segmentation_results(seg_path, img_path)