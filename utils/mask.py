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


def create_masks(input_dir, output_dir):
    """
    将分割图像转换为天空、刚性和非刚性物体的掩码
    :param input_dir: 输入图像目录
    :param output_dir: 输出掩码目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建三个输出子目录
    subdirs = ['sky', 'rigid', 'nonrigid']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    seg_files = [f for f in os.listdir(input_dir) if f.endswith('_camera_seg_0.png') or 
                 f.endswith('_camera_seg_1.png') or 
                 f.endswith('_camera_seg_2.png')]
    print(f"开始处理 {len(seg_files)} 张分割图像...")

    for filename in tqdm(seg_files, desc="处理进度"):
        if filename.endswith('_camera_seg_0.png') or filename.endswith('_camera_seg_1.png') or filename.endswith('_camera_seg_2.png'):
            file_path = os.path.join(input_dir, filename)
            # 提取基础文件名和后缀编号
            parts = filename.split('_camera_seg')
            base_name = parts[0]
            suffix = parts[1].split('.')[0]  # 获取后缀编号（空字符串、_1、_2）
            
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

    print("所有图像处理完成！")
    
def apply_mask(image_dir, mask_dir, output_dir):
    """
    将原图像与mask叠加，输出mask之后的图像
    :param image_dir: 原图像目录
    :param mask_dir: mask图像目录
    :param output_dir: 输出图像目录
    """
    # 定义三种mask类型
    mask_types = ['sky', 'rigid', 'nonrigid']
    
    # 为每种mask类型创建输出子目录
    output_dirs = {}
    for mask_type in mask_types:
        output_dirs[mask_type] = os.path.join(output_dir, mask_type)
        os.makedirs(output_dirs[mask_type], exist_ok=True)
    
    # 获取所有原图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and '_camera_' in f]
    print(f"开始处理 {len(image_files)} 张原图像...")

    for filename in tqdm(image_files, desc="处理进度"):
        # 解析文件名
        parts = filename.split('_camera_')
        frame_num = parts[0]  # 帧数部分，如000xxx
        camera_num = parts[1].split('.')[0]  # 相机编号，如0, 1, 2

        # 读取原图像
        image_path = os.path.join(image_dir, filename)
        image = np.array(Image.open(image_path))

        # 如果原图像是RGBA格式，只取前三个通道
        if image.shape[2] == 4:
            image = image[..., :3]

        # 处理每种mask类型
        for mask_type in mask_types:
            # 构建对应的mask文件名
            mask_filename = f"{frame_num}_{mask_type}_{camera_num}.png"
            mask_path = os.path.join(mask_dir, mask_type, mask_filename)

            if not os.path.exists(mask_path):
                # print(f"警告：未找到对应的{mask_type} mask文件 {mask_filename}")
                continue

            # 读取mask图像
            mask = np.array(Image.open(mask_path))

            # 确保mask是单通道
            if mask.ndim == 3:
                mask = mask[..., 0]  # 取第一个通道

            # 将mask应用到原图像
            masked_image = np.where(mask[..., None] == 255, image, 0)

            # 保存结果到对应的子目录
            output_path = os.path.join(output_dirs[mask_type], 
                                     f"{frame_num}_masked_{mask_type}_camera_{camera_num}.png")
            Image.fromarray(masked_image).save(output_path)

    print("所有图像处理完成！")


if __name__ == "__main__":
    # 定义基础路径
    base_dir = '../data/training_20250226_102047'
    
    # 基于base路径定义其他路径
    image_dir = f'{base_dir}/image'
    mask_dir = f'{base_dir}/mask'
    output_dir = f'{base_dir}/masked_images'
    
    # 可以选择处理哪种mask类型
    # create_masks(image_dir, mask_dir)
    apply_mask(image_dir, mask_dir, output_dir)