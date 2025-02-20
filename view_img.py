import cv2
import os

# 图片目录
image_dir = 'data/training/bbox_img'

# 获取所有camera_0的图片
# images = sorted([img for img in os.listdir(image_dir) if img.endswith('_camera_0.png')])
images = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
# 遍历并显示图片
for img_name in images:
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图片: {img_path}")
        continue
        
    cv2.imshow('Camera 0 Sequence', img)
    
    # 按任意键继续，按'q'退出
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()