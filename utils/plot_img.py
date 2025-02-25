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
rigid_ids = [14, 15, 16, 17, 18, 19]  # 汽车、卡车、公交车等
nonrigid_ids = [12, 13]  # 行人、骑行者

# 使用PIL读取图像
img = Image.open('../data/training/image/000000_camera_seg.png')
img_array = np.array(img)

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

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(sky_img)
plt.title('Sky')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(rigid_img)
plt.title('Rigid Dynamic')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(nonrigid_img)
plt.title('Nonrigid')
plt.axis('off')

plt.tight_layout()
plt.show()