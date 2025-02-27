from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
all_ids = [11, 12, 13, 18, 19, 14, 15, 16, 17]

# 使用PIL读取图像
img = Image.open('../data/training_20250226_140258/image/000010_camera_seg_0.png')
img_array = np.array(img)

from skimage.segmentation import flood_fill

def get_ego_mask(img_array):
    # 将输入转换为数值类型（原mask是布尔类型）
    working_array = img_array.astype(np.uint8) * 255  # 转换为0和255的数值
    height, width = working_array.shape[:2]  # 处理二维数组
    
    # 调整种子点坐标（二维数组不需要通道维度）
    seed_point = (height - 1, width // 2)
    
    # 使用单通道数值进行填充
    ego_mask = flood_fill(working_array, seed_point, 0, tolerance=10)
    return ego_mask == 0  # 返回布尔mask

# 获取ego mask
# ego_mask = get_ego_mask(img_array)

# 创建三个新的RGB图像
height, width, _ = img_array.shape
sky_img = np.zeros((height, width, 3), dtype=np.uint8)
rigid_img = np.zeros((height, width, 3), dtype=np.uint8)
nonrigid_img = np.zeros((height, width, 3), dtype=np.uint8)
all_img = np.zeros((height, width, 3), dtype=np.uint8)

# 处理天空图像
for id in sky_ids:
    mask = img_array[:, :, 0] == id
    sky_img[mask] = (255, 255, 255)  # 白色

# 处理刚性物体图像
for id in rigid_ids:
    mask = img_array[:, :, 0] == id
    # print(mask.shape)
    ego_mask = get_ego_mask(mask)
    mask[ego_mask] = False  # 排除ego车辆区域
    rigid_img[mask] = (255, 255, 255)  # 白色

# 处理非刚性物体图像
for id in nonrigid_ids:
    mask = img_array[:, :, 0] == id
    nonrigid_img[mask] = (255, 255, 255)  # 白色
    


# 创建动态物体图像（合并刚性和非刚性）
dynamic_img = np.zeros((height, width, 3), dtype=np.uint8)
dynamic_img = np.logical_or(rigid_img, nonrigid_img) * 255

# 获取全局ego mask（使用分割图的第一个通道）
ego_mask = get_ego_mask(img_array[:, :, 0])

# 创建新的绘图布局（2x3网格）
plt.figure(figsize=(15, 8))  # 调整画布尺寸适应横向布局

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(Image.open('../data/training_20250226_140258/image/000010_camera_0.png'))
plt.title('Original Image')
plt.axis('off')

# 分割原图
plt.subplot(2, 3, 2)
plt.imshow(dynamic_img)
plt.title('Dynamic')
plt.axis('off')

# 天空区域
plt.subplot(2, 3, 3)
plt.imshow(sky_img)
plt.title('Sky')
plt.axis('off')

# 刚性动态物体
plt.subplot(2, 3, 4)
plt.imshow(rigid_img)
plt.title('Rigid Dynamic')
plt.axis('off')

# 非刚性动态物体
plt.subplot(2, 3, 5)
plt.imshow(nonrigid_img)
plt.title('Nonrigid')
plt.axis('off')

# Ego Mask
plt.subplot(2, 3, 6)
plt.imshow(ego_mask, cmap='gray')
plt.title('Ego Vehicle Mask')
plt.axis('off')

plt.tight_layout()
plt.show()