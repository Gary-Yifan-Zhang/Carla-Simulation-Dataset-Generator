import open3d as o3d
import numpy as np

def read_point_cloud(file_path):
    # 读取二进制点云文件
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # 只取前三列（x, y, z）

def read_bounding_boxes(file_path):
    # 读取边界框文件
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            # 假设数据格式为：type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y
            object_type = data[0]  # 获取对象类型
            h, w, l = float(data[8]), float(data[9]), float(data[10])
            x, y, z = float(data[11]), float(data[12]), float(data[13])
            rotation_y = float(data[14])
            bbox = create_bbox(x, y, z, h, w, l, rotation_y, object_type)  # 传递对象类型
            bboxes.append(bbox)
    return bboxes

def create_bbox(x, y, z, h, w, l, rotation_y, object_type):
    # 创建边界框
    bbox = o3d.geometry.OrientedBoundingBox()
    
    if object_type == "Pedestrian":  # 行人
        bbox.center = [x, y, z]  # 使用原始中心
        bbox.extent = [h, w, l]
        print(rotation_y)
        print("Pedestrian")
    elif object_type == "Car":  # 车辆
        bbox.center = [x, y, h/2+0.32]  # 底部中心，z=0
        bbox.extent = [h, w, l]
        print(rotation_y)
        print("Car")
    
    
    # 设置旋转
    R = np.array([
            [np.cos(rotation_y), -np.sin(rotation_y), 0],
            [np.sin(rotation_y), np.cos(rotation_y), 0],
            [0, 0, 1]
        ])
    r_velo_to_cam = np.array([[0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [1.0, 0.0, 0.0],
                                ])
    R = R @ r_velo_to_cam
    # R = r_velo_to_cam @ R
    bbox.rotate(R, center=bbox.center)
    return bbox

def visualize(point_cloud, bboxes):
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 设置边界框的颜色
    for bbox in bboxes:
        bbox.color = (1, 0, 0)  # 红色

    # 可视化
    o3d.visualization.draw_geometries([pcd] + bboxes)

if __name__ == "__main__":
    point_cloud = read_point_cloud("data/training/velodyne/000112.bin")
    bboxes = read_bounding_boxes("data/training/lidar_label/000112.txt")
    visualize(point_cloud, bboxes)
    print("Done")
