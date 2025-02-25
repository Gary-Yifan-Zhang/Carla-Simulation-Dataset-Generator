import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# 定义类别
id_to_color = {
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
sky_ids = [11]
rigid_ids = [14, 15, 16, 17]  # 汽车、卡车、公交车等
nonrigid_ids = [12, 13, 18, 19]  # 行人、骑行者

# 输入输出路径配置
input_dir = '../data/training_20250225_151840/image'  # 原始seg图像目录
output_dir = '../data/training_20250225_151840/mask'  # 修改后的输出路径
os.makedirs(output_dir, exist_ok=True)


# 创建三个输出子目录
subdirs = ['sky', 'rigid', 'nonrigid']
for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

seg_files = [f for f in os.listdir(input_dir) if f.endswith('_camera_seg.png') or 
             f.endswith('_camera_seg_1.png') or 
             f.endswith('_camera_seg_2.png')]
print(f"开始处理 {len(seg_files)} 张分割图像...")

# 修改文件名处理逻辑
for filename in tqdm(seg_files, desc="处理进度"):
    if filename.endswith('_camera_seg.png') or filename.endswith('_camera_seg_1.png') or filename.endswith('_camera_seg_2.png'):
        file_path = os.path.join(input_dir, filename)
        # 提取基础文件名和后缀编号
        parts = filename.split('_camera_seg')
        base_name = parts[0]
        suffix = parts[1].split('.')[0]  # 获取后缀编号（空字符串、_1、_2）
        
        base_name = filename.split('_camera_seg')[0]
        
        # 读取并处理图像
        img_array = np.array(Image.open(file_path))
        
        # 创建三个新的RGB图像
        height, width, _ = img_array.shape
        sky_img = np.zeros((height, width, 3), dtype=np.uint8)
        rigid_img = np.zeros((height, width, 3), dtype=np.uint8)
        nonrigid_img = np.zeros((height, width, 3), dtype=np.uint8)

        # 处理天空图像
        for id in sky_ids:
            mask = img_array[:, :, 0] == id
            sky_img[mask] = (255, 255, 255)  # 白色

        # 处理刚性物体图像
        for id in rigid_ids:
            mask = img_array[:, :, 0] == id
            rigid_img[mask] = (255, 255, 255)  # 白色

        # 处理非刚性物体图像
        for id in nonrigid_ids:
            mask = img_array[:, :, 0] == id
            nonrigid_img[mask] = (255, 255, 255)  # 白色
        
        # 保存处理结果
        Image.fromarray(sky_img).save(f'{output_dir}/sky/{base_name}_sky{suffix}.png')
        Image.fromarray(rigid_img).save(f'{output_dir}/rigid/{base_name}_rigid{suffix}.png')
        Image.fromarray(nonrigid_img).save(f'{output_dir}/nonrigid/{base_name}_nonrigid{suffix}.png')
