import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.segmentation import flood_fill  # 新增导入
import cv2


def get_ego_mask(mask_array):
    """生成ego车辆区域的掩码"""
    working_array = mask_array.astype(np.uint8) * 255
    height, width = working_array.shape
    seed_point = (height - 1, width // 2)  # 底部中心作为种子点
    ego_mask = flood_fill(working_array, seed_point, 0, tolerance=10)
    return ego_mask == 0


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

def get_segmentation_files(input_dir, camera_count=5):
    """获取所有相机的分割图像文件"""
    return [f for f in os.listdir(input_dir) 
           if any(f.endswith(f'_camera_seg_{i}.png') for i in range(camera_count))]


def create_masks(input_dir, output_dir):
    """
    将分割图像转换为天空、刚性和非刚性物体的掩码
    :param input_dir: 输入图像目录
    :param output_dir: 输出掩码目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建四个输出子目录（新增ego目录）
    subdirs = ['sky', 'rigid', 'nonrigid', 'ego']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # 新增：记录已保存的相机ego mask
    processed_cameras = set()

    seg_files = get_segmentation_files(input_dir, camera_count=5)  # 假设现在有5个相机
    print(f"开始处理 {len(seg_files)} 张分割图像（共{5}个相机）...")


    for filename in tqdm(seg_files, desc="处理进度"):
        if '_camera_seg_' in filename:
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

            parts = filename.split('_')
            camera_id = int(parts[-1].split('.')[0])  # 获取最后的数字作为相机ID

            # 处理天空图像
            for id in sky_ids:
                mask = img_array[:, :, 0] == id
                sky_img[mask] = (255, 255, 255)  # 白色

            # 处理刚性物体图像
            for id in rigid_ids:
                mask = img_array[:, :, 0] == id
                # 新增ego mask排除逻辑
                ego_mask = get_ego_mask(mask)

                mask[ego_mask] = False  # 排除ego车辆区域
                rigid_img[mask] = (255, 255, 255)

            # 处理非刚性物体图像
            for id in nonrigid_ids:
                mask = img_array[:, :, 0] == id
                nonrigid_img[mask] = (255, 255, 255)  # 白色
            
            # 新增：生成ego mask
            ego_mask = get_ego_mask(img_array[:, :, 0])
            ego_img = np.zeros((height, width, 3), dtype=np.uint8)
            ego_img[ego_mask] = (255, 255, 255)  # 白色表示ego区域


            # 保存处理结果
            Image.fromarray(sky_img).save(f'{output_dir}/sky/{base_name}_sky{suffix}.png')
            Image.fromarray(rigid_img).save(f'{output_dir}/rigid/{base_name}_rigid{suffix}.png')
            Image.fromarray(nonrigid_img).save(f'{output_dir}/nonrigid/{base_name}_nonrigid{suffix}.png')
            Image.fromarray(ego_img).save(f'{output_dir}/ego/{base_name}_ego{suffix}.png')  # 新增行


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
    image_files = [f for f in os.listdir(
        image_dir) if f.endswith('.png') and '_camera_' in f]
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


def generate_bbox_masks(label_dir, output_dir, **kwargs):
    """
    更新后的边界框掩码生成函数（带路径参数）
    :param label_dir: 标注文件目录（必须参数）
    :param output_dir: 输出目录（必须参数）
    :param kwargs: 可选参数 width/height/show/save
    """
    # 从kwargs获取参数或使用默认值
    width = kwargs.get('width', 1920)
    height = kwargs.get('height', 1080)
    show = kwargs.get('show', False)
    save = kwargs.get('save', True)
    # 创建输出目录（如果不存在）
    if save:
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有标注文件（保持与create_masks相同的文件过滤逻辑）
    seg_files = [f for f in os.listdir(
        label_dir) if f.endswith('.txt') and '_camera_' in f]
    print(f"开始处理 {len(seg_files)} 个标注文件...")

    for filename in tqdm(seg_files, desc="生成边界框掩码"):
        file_path = os.path.join(label_dir, filename)

        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(label_dir, filename)
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []

        # 第一次读取：生成掩码
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue

                    left = int(float(parts[5]))
                    top = int(float(parts[6]))
                    right = int(float(parts[7]))
                    bottom = int(float(parts[8]))

                    # 坐标安全限制
                    left = max(0, min(left, width-1))
                    right = max(0, min(right, width-1))
                    top = max(0, min(top, height-1))
                    bottom = max(0, min(bottom, height-1))

                    mask[top:bottom, left:right] = 1
                    bboxes.append((left, top, right, bottom))

        # 可视化调试
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 6))

            # 掩码可视化
            plt.subplot(1, 2, 1)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask Preview: {filename}')

            # 边界框可视化
            plt.subplot(1, 2, 2)
            fake_img = np.zeros((height, width, 3), dtype=np.uint8)
            for box in bboxes:
                cv2.rectangle(fake_img,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 255, 0), 2)
            plt.imshow(fake_img)
            plt.title(f'BBoxes: {len(bboxes)} objects')
            plt.tight_layout()
            plt.show(block=True)

        # 保存结果
        if save:
            mask_name = filename.replace(
                '.txt', '_mask.png').replace('_camera_', '_bbox_')
            output_path = os.path.join(output_dir, mask_name)
            Image.fromarray(mask * 255).save(output_path)

        else:
            print(f'Processed: {filename} (not saved)')


def combine_masks(base_dir, **kwargs):
    """
    更新后的组合掩码函数
    :param base_dir: 数据集根目录（必须参数）
    :param kwargs: 可选参数 label_subdir/mask_subdirs/output_subdir
    """
    # 从kwargs获取参数并设置默认值
    label_subdir = kwargs.get('label_subdir', 'image_label')
    mask_subdirs = kwargs.get('mask_subdirs', ('mask/rigid', 'mask/nonrigid'))
    output_subdir = kwargs.get('output_subdir', 'mask/object_intersection')
    
    # 路径配置
    label_dir = os.path.join(base_dir, label_subdir)
    output_root = os.path.join(base_dir, output_subdir)
    os.makedirs(output_root, exist_ok=True)

    # 创建类别子目录
    for mask_type in mask_subdirs:
        type_dir = os.path.join(output_root, mask_type.split('/')[-1])
        os.makedirs(type_dir, exist_ok=True)

    # 获取标注文件（保持原有过滤逻辑）
    seg_files = [f for f in os.listdir(label_dir)
                 if f.endswith('.txt') and '_camera_' in f]

    print(f"开始处理 {len(seg_files)} 个文件...")

    for filename in tqdm(seg_files, desc="合成刚体/非刚体"):
        # 解析文件名
        base_name = filename.replace('.txt', '')
        frame_num, camera_id = base_name.split('_camera_')

        # 生成bbox掩码（复用原有安全机制）
        bbox_mask = np.zeros((1080, 1920), dtype=np.uint8)
        with open(os.path.join(label_dir, filename), 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    left, top = map(lambda x: max(
                        0, int(float(x))), (parts[5], parts[6]))
                    right, bottom = map(lambda x: min(
                        1920, int(float(x))), (parts[7], parts[8]))
                    bbox_mask[top:bottom, left:right] = 255

        # 分别处理两类掩码
        for mask_type in mask_subdirs:
            type_name = mask_type.split('/')[-1]  # rigid/nonrigid
            input_path = os.path.join(base_dir, mask_type,
                                      f"{frame_num}_{type_name}_{camera_id}.png")
            output_path = os.path.join(output_root, type_name,
                                       f"{frame_num}_intersection_{type_name}_{camera_id}.png")

            if not os.path.exists(input_path):
                continue

            # 处理掩码交集
            original_mask = np.array(Image.open(input_path).convert('L'))
            _, binary_mask = cv2.threshold(
                original_mask, 1, 255, cv2.THRESH_BINARY)
            intersection = cv2.bitwise_and(binary_mask, bbox_mask)

            # 保存优化后的掩码
            Image.fromarray(intersection).save(output_path, compress_level=9)

    print("刚体/非刚体掩码分离保存完成！")
    
def process_all_masks(base_dir):
    """
    完整的mask处理流水线
    参数:
        base_dir: 数据集根目录 (包含image/和image_label/)
    """
    # 定义标准路径
    image_dir = os.path.join(base_dir, "image")
    label_dir = os.path.join(base_dir, "image_label")
    mask_dir = os.path.join(base_dir, "mask")
    masked_images_dir = os.path.join(base_dir, "masked_images")
    bbox_dir = os.path.join(mask_dir, "bbox")
    
    # 按顺序执行处理流程
    print("\n" + "="*40)
    print("开始生成基础语义分割掩码")
    create_masks(image_dir, mask_dir)
    
    print("\n" + "="*40)
    print("生成叠加mask的可视化图像")
    apply_mask(image_dir, mask_dir, masked_images_dir)
    
    print("\n" + "="*40)
    print("生成边界框掩码")
    generate_bbox_masks(
        label_dir=label_dir,
        output_dir=bbox_dir,
        width=1920,  # 从配置读取或保持默认
        height=1080
    )
    
    print("\n" + "="*40)
    print("生成组合掩码")
    combine_masks(
        base_dir=base_dir,
        label_subdir="image_label",
        mask_subdirs=('mask/rigid', 'mask/nonrigid'),
        output_subdir="mask/object_intersection"
    )



if __name__ == "__main__":
    # 定义基础路径
    base_dir = '../data/training_20250226_102047'

    # 基于base路径定义其他路径
    image_dir = f'{base_dir}/image'
    mask_dir = f'{base_dir}/mask'
    output_dir = f'{base_dir}/masked_images'

    # # 可以选择处理哪种mask类型
    # create_masks(image_dir, mask_dir)
    # apply_mask(image_dir, mask_dir, output_dir)
    # generate_bbox_masks(
    # label_dir='../data/training_20250226_102047/image_label',
    # output_dir='../data/training_20250226_102047/mask/bbox'
    # )
    combine_masks()
