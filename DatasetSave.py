from utils.save import *
from utils.utils import config_transform_to_carla_transform
import time


class DatasetSave:
    def __init__(self, config):
        """
            初始化

            参数：
                config：预设配置
        """
        self.config = config
        self.OUTPUT_FOLDER = None
        self.EGO_STATE_PATH = None

        self.CALIBRATION_PATH = None

        self.IMAGE_PATH = None
        self.IMAGE_LABEL_PATH = None
        self.BBOX_IMAGE_PATH = None

        self.LIDAR_PATH = None
        self.LIDAR_LABEL_PATH = None
        
        self.DEPTH_PATH = None 
        self.SEMANTIC_PATH = None

        self.generate_path(self.config["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self.get_current_files_num()

    def generate_path(self, root_path):
        """
            生成数据存储的路径

            参数：
                root_path：根目录的路径
        """

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(timestamp)
        PHASE = f"training_{timestamp}"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ['calib', 'image', 'image_label', 'bbox_img', 
              'velodyne', 'lidar_label', 'ego_state', 'extrinsic', 
              'depth', 'semantic']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')

        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}_camera_{1}.png')
        self.IMAGE_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_label/{0:06}.txt')
        self.BBOX_IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'bbox_img/{0:06}.png')

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}_lidar_{1}.bin')
        self.LIDAR_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'lidar_label/{0:06}.txt')
        self.EGO_STATE_PATH = os.path.join(self.OUTPUT_FOLDER, 'ego_state/{0:06}.txt')
        self.EXTRINSIC_PATH = os.path.join(self.OUTPUT_FOLDER, 'extrinsic/{{id}}.txt')
        self.GLOBEL_EXTRINSIC_PATH = os.path.join(self.OUTPUT_FOLDER, 'extrinsic/{0:06}.npz')
        self.EXTRINSIC_TXT_PATH = os.path.join(self.OUTPUT_FOLDER, 'extrinsic/{0:06}.txt')
        self.DEPTH_PATH = os.path.join(self.OUTPUT_FOLDER, 'depth/{0:06}_depth_{1}.png')
        self.SEMANTIC_PATH = os.path.join(self.OUTPUT_FOLDER, 'semantic/{0:06}_semantic_{1}.png')



    def get_current_files_num(self):
        """
            获取文件夹中存在的数据量

            返回：
                num_existing_data_files：文件夹内存在的数据量
        """
        label_path = os.path.join(self.OUTPUT_FOLDER, 'image_label/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print("当前存在{}个数据".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def save_datasets(self, data):
        """
            保存数据集

            返回：
                data：CARLA传感器相关数据（原始数据，内参，外参等）
        """
        # 路径格式定义保持不变
        self.IMAGE_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_label/{0:06}_camera_{1}.txt')
        self.BBOX_IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'bbox_img/{0:06}_camera_{1}.png')

        # 使用列表推导式生成多相机文件名
        camera_count = 5
        img_label_filenames = [
            self.IMAGE_LABEL_PATH.format(self.captured_frame_no, i)
            for i in range(camera_count)
        ]
        bbox_img_filenames = [
            self.BBOX_IMAGE_PATH.format(self.captured_frame_no, i)
            for i in range(camera_count)
        ]

        # 生成带特殊后缀的图像路径
        special_images = {
            'view': self.IMAGE_PATH.format(self.captured_frame_no, "view"),
            'bev': self.IMAGE_PATH.format(self.captured_frame_no, "bev"),
            **{
                f'seg_{i}': self.IMAGE_PATH.format(self.captured_frame_no, f"seg_{i}")
                for i in range(5)
            }
        }

        # 定义传感器到文件编号的映射
        sensor_mapping = {
            "RGB": "000",
            "SUB_RGB_1": "001", 
            "SUB_RGB_2": "002"
        }
        
        # 保存外参文件
        base_extrinsic_path = self.EXTRINSIC_PATH.format(self.captured_frame_no)
        save_extrinsic_matrices(self.config, base_extrinsic_path, sensor_mapping)

        # 传感器数据索引映射（保持原始索引对应关系）
        sensor_mapping = [
            (0, 0),    # camera 0 -> sensor_data[0]
            (1, 4),    # camera 1 -> sensor_data[4]
            (2, 5),    # camera 2 -> sensor_data[5]
            (3, 13),   # camera 3 -> sensor_data[13]
            (4, 14)    # camera 4 -> sensor_data[14]
        ]

        # 生成基础图像路径
        base_images = [
            (self.IMAGE_PATH.format(self.captured_frame_no, cam_idx), data_idx)
            for cam_idx, data_idx in sensor_mapping
        ]

        # 生成深度和语义路径（保持原始索引）
        depth_files = [
            self.DEPTH_PATH.format(self.captured_frame_no, i)
            for i, idx in enumerate([0, 1, 2])
        ]
        semantic_files = [
            self.SEMANTIC_PATH.format(self.captured_frame_no, i)
            for i, idx in enumerate([0, 1, 2])
        ]

        # 激光雷达路径生成（保持原始格式）
        lidar_files = [
            self.LIDAR_PATH.format(self.captured_frame_no, i)
            for i in range(5)
        ]

        # 其他路径保持不变
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        lidar_label_filename = self.LIDAR_LABEL_PATH.format(self.captured_frame_no)
        ego_state_filename = self.EGO_STATE_PATH.format(self.captured_frame_no)

        for agent, dt in data["agents_data"].items():
            extrinsic = dt["extrinsic"]
            # 外参处理保持不变
            extrinsic_dict = {
                "RGB": extrinsic[0],
                "SUB_RGB_1": extrinsic[4],
                "SUB_RGB_2": extrinsic[5],
                "LIDAR": extrinsic[2]
            }
            save_globel_extrinsic_matrices(
                self.GLOBEL_EXTRINSIC_PATH.format(self.captured_frame_no),
                sensor_mapping,
                extrinsic_dict
            )

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
            save_calibration_matrices(extrinsic, calib_filename, dt["intrinsic"])

            # 保存基础摄像头图像
            for img_path, data_idx in base_images:
                save_image_data(img_path, dt["sensor_data"][data_idx])

            # 保存分割图像
            for seg_cam in [0, 1, 2, 3, 4]:
                save_image_data(
                    special_images[f'seg_{seg_cam}'],
                    dt["sensor_data"][[8, 9, 10, 15, 16][seg_cam]]
                )

            # 激光雷达数据保存（保持原始TODO注释）
            save_lidar_data(lidar_files[0], dt["sensor_data"][2], extrinsic[2])
            # save_lidar_data(lidar_files[1], dt["sensor_data"][6], extrinsic[6])  # 保持注释状态
            save_kitti_label_data(lidar_label_filename, dt["pc_labels_kitti"])

            # 保存标签和标注数据
            for cam_idx in range(camera_count):
                save_kitti_label_data(
                    img_label_filenames[cam_idx],
                    dt[f"image_labels_kitti{'_'+str(cam_idx) if cam_idx>0 else ''}"]
                )
                save_bbox_image_data(
                    bbox_img_filenames[cam_idx],
                    dt[f"bbox_img{'_'+str(cam_idx) if cam_idx>0 else ''}"]
                )
            # EGO状态保存保持不变
            save_ego_data(
                ego_state_filename,
                transform=data["egostate"]["location"],
                rotation=data["egostate"]["rotation"],
                velocity=data["egostate"]["velocity"],
                acceleration=data["egostate"]["acceleration"],
                extent=data["egostate"]["extent"]
            )
                        
            # 保存深度和语义数据
            for i, depth_idx in enumerate([1, 11, 12]):
                save_depth_image_data(depth_files[i], dt["sensor_data"][depth_idx])
            for i, seg_idx in enumerate([8, 9, 10]):
                save_semantic_image_data(semantic_files[i], dt["sensor_data"][seg_idx])

        self.captured_frame_no += 1
