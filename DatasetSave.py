from utils.save import *
from utils.utils import config_transform_to_carla_transform


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

        self.generate_path(self.config["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self.get_current_files_num()

    def generate_path(self, root_path):
        """
            生成数据存储的路径

            参数：
                root_path：根目录的路径
        """
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ['calib', 'image', 'image_label', 'bbox_img', 'velodyne', 'lidar_label', 'ego_state']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')

        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}_camera_{1}.png')
        self.IMAGE_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_label/{0:06}.txt')
        self.BBOX_IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'bbox_img/{0:06}.png')

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.LIDAR_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'lidar_label/{0:06}.txt')
        self.EGO_STATE_PATH = os.path.join(self.OUTPUT_FOLDER, 'ego_state/{0:06}.txt')

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
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)

        img_filename_0 = self.IMAGE_PATH.format(self.captured_frame_no, 0)
        img_filename_4 = self.IMAGE_PATH.format(self.captured_frame_no, 4)
        img_filename_5 = self.IMAGE_PATH.format(self.captured_frame_no, 5)
        img_label_filename = self.IMAGE_LABEL_PATH.format(self.captured_frame_no)
        bbox_img_filename = self.BBOX_IMAGE_PATH.format(self.captured_frame_no)

        lidar_filename = self.LIDAR_PATH.format(self.captured_frame_no)
        lidar_label_filename = self.LIDAR_LABEL_PATH.format(self.captured_frame_no)

        ego_state_filename = os.path.join(self.EGO_STATE_PATH.format(self.captured_frame_no))

        for agent, dt in data["agents_data"].items():
            extrinsic = dt["extrinsic"]

            camera_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)

            save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
            save_image_data(img_filename_0, dt["sensor_data"][0])
            save_image_data(img_filename_4, dt["sensor_data"][4])
            save_image_data(img_filename_5, dt["sensor_data"][5])
            save_kitti_label_data(img_label_filename, dt["image_labels_kitti"])
            save_bbox_image_data(bbox_img_filename, dt["bbox_img"])

            save_lidar_data(lidar_filename, dt["sensor_data"][2], extrinsic)
            save_kitti_label_data(lidar_label_filename, dt["pc_labels_kitti"])

            save_ego_data(ego_state_filename, dt["transform"], dt["velocity"], dt["acceleration"])

        self.captured_frame_no += 1
