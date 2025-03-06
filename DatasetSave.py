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
              'velodyne', 'lidar_label', 'ego_state', 'extrinsic', 'depth']

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
        self.DEPTH_PATH = os.path.join(self.OUTPUT_FOLDER, 'depth/{0:06}_depth_{1}.png')


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
        
        # 修改图像标签和标注图像路径格式
        self.IMAGE_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_label/{0:06}_camera_{1}.txt')
        self.BBOX_IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'bbox_img/{0:06}_camera_{1}.png')

        # 生成三个摄像头的文件名
        img_label_filename_0 = self.IMAGE_LABEL_PATH.format(self.captured_frame_no, 0)
        img_label_filename_1 = self.IMAGE_LABEL_PATH.format(self.captured_frame_no, 1)
        img_label_filename_2 = self.IMAGE_LABEL_PATH.format(self.captured_frame_no, 2)
        
        bbox_img_filename_0 = self.BBOX_IMAGE_PATH.format(self.captured_frame_no, 0)
        bbox_img_filename_1 = self.BBOX_IMAGE_PATH.format(self.captured_frame_no, 1)
        bbox_img_filename_2 = self.BBOX_IMAGE_PATH.format(self.captured_frame_no, 2)

        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)

        img_filename_0 = self.IMAGE_PATH.format(self.captured_frame_no, 0)
        img_filename_1 = self.IMAGE_PATH.format(self.captured_frame_no, 1)
        img_filename_2 = self.IMAGE_PATH.format(self.captured_frame_no, 2)
        img_filename_view = self.IMAGE_PATH.format(self.captured_frame_no, "view")
        img_filename_bev = self.IMAGE_PATH.format(self.captured_frame_no, "bev")
        img_filename_seg = self.IMAGE_PATH.format(self.captured_frame_no, "seg_0")
        img_filename_seg_1 = self.IMAGE_PATH.format(self.captured_frame_no, "seg_1")
        img_filename_seg_2 = self.IMAGE_PATH.format(self.captured_frame_no, "seg_2")
        # img_label_filename = self.IMAGE_LABEL_PATH.format(self.captured_frame_no)
        # bbox_img_filename = self.BBOX_IMAGE_PATH.format(self.captured_frame_no)
        
        depth_filename_0 = self.DEPTH_PATH.format(self.captured_frame_no, 0)
        depth_filename_1 = self.DEPTH_PATH.format(self.captured_frame_no, 1)
        depth_filename_2 = self.DEPTH_PATH.format(self.captured_frame_no, 2)


        lidar_filename = self.LIDAR_PATH.format(self.captured_frame_no, 0)
        lidar_filename_1 = self.LIDAR_PATH.format(self.captured_frame_no, 1)
        lidar_filename_2 = self.LIDAR_PATH.format(self.captured_frame_no, 2)
        lidar_filename_3 = self.LIDAR_PATH.format(self.captured_frame_no, 3)
        lidar_filename_4 = self.LIDAR_PATH.format(self.captured_frame_no, 4)

        lidar_label_filename = self.LIDAR_LABEL_PATH.format(self.captured_frame_no)

        ego_state_filename = os.path.join(self.EGO_STATE_PATH.format(self.captured_frame_no))
        
        # 定义传感器到文件编号的映射
        sensor_mapping = {
            "RGB": "000",
            "SUB_RGB_1": "001", 
            "SUB_RGB_2": "002"
        }
        
        # 保存外参文件
        base_extrinsic_path = self.EXTRINSIC_PATH.format(self.captured_frame_no)
        save_extrinsic_matrices(self.config, base_extrinsic_path, sensor_mapping)

        for agent, dt in data["agents_data"].items():
            extrinsic = dt["extrinsic"]

            camera_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)

            save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
            save_image_data(img_filename_0, dt["sensor_data"][0])
            save_image_data(img_filename_1, dt["sensor_data"][4])
            save_image_data(img_filename_2, dt["sensor_data"][5])
            save_image_data(img_filename_seg, dt["sensor_data"][8])
            save_image_data(img_filename_seg_1, dt["sensor_data"][9])
            save_image_data(img_filename_seg_2, dt["sensor_data"][10])
            # save_image_data(img_filename_view,dt["sensor_data"][10])
            # save_image_data(img_filename_bev,dt["sensor_data"][11])

            
            # 保存三个摄像头的标签和标注图像
            save_kitti_label_data(img_label_filename_0, dt["image_labels_kitti"])
            save_kitti_label_data(img_label_filename_1, dt["image_labels_kitti_1"])
            save_kitti_label_data(img_label_filename_2, dt["image_labels_kitti_2"])
            
            save_bbox_image_data(bbox_img_filename_0, dt["bbox_img"])
            save_bbox_image_data(bbox_img_filename_1, dt["bbox_img_1"])
            save_bbox_image_data(bbox_img_filename_2, dt["bbox_img_2"])
            
            save_depth_image_data(depth_filename_0, dt["sensor_data"][1])
            save_depth_image_data(depth_filename_1, dt["sensor_data"][11])
            save_depth_image_data(depth_filename_2, dt["sensor_data"][12])
            


            #TODO: 修改为保存多个雷达数据
            save_lidar_data(lidar_filename, dt["sensor_data"][2], extrinsic[2])
            # save_lidar_data(lidar_filename_1, dt["sensor_data"][6], extrinsic[6])
            # save_lidar_data(lidar_filename_2, dt["sensor_data"][7], extrinsic)
            # save_lidar_data(lidar_filename_3, dt["sensor_data"][8], extrinsic)
            # save_lidar_data(lidar_filename_4, dt["sensor_data"][9], extrinsic)
            save_kitti_label_data(lidar_label_filename, dt["pc_labels_kitti"])

            save_ego_data(ego_state_filename, dt["transform"], dt["velocity"], dt["acceleration"])

        self.captured_frame_no += 1
