import carla
import random
import logging
import queue
import numpy as np
from utils.utils import config_transform_to_carla_transform, set_camera_intrinsic, object_filter_by_distance
from utils.label import spawn_dataset
from scenario_runner import ScenarioRunner

class SimulationScene:
    def __init__(self, config):
        """
            初始化

            参数：
                config：预设配置
        """
        self.config = config
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        # 设置Carla地图
        self.world = None
        self.traffic_manager = self.client.get_trafficmanager()
        self.init_settings = None
        self.frame = None
        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}, "environment_data": None}  # 记录每一帧的数据
        self.vehicle = None
        self.agent_transform = None
        self.agent = None
        # 新增随机种子配置
        self.random_seed = 42  # 固定随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def set_map(self):
        """
            设置场景地图
        """
        self.world = self.client.load_world('Town02')

    def set_weather(self):
        """
            设置场景天气
        """
        weather = carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=50.0)
        self.world.set_weather(weather)

    def set_synchrony(self):
        """
            开启同步模式
        """
        self.init_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # 固定时间步长 (0.05s, 20fps)
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def spawn_actors(self):
        """
            在场景中生成actors(车辆与行人)
        """
        # 生成车辆
        num_of_vehicles = self.config["CARLA_CONFIG"]["NUM_OF_VEHICLES"]
        blueprints = sorted(self.world.get_blueprint_library().filter("vehicle.*"), key=lambda bp: bp.id)
        spawn_points = sorted(self.world.get_map().get_spawn_points(), key=lambda x: str(x.location))  # 固定生成点顺序
        number_of_spawn_points = len(spawn_points)

        if num_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
            num_of_vehicles = num_of_vehicles
        elif num_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, number_of_spawn_points)
            num_of_vehicles = number_of_spawn_points

        batch = []
        for n, transform in enumerate(spawn_points[:num_of_vehicles]):
            blueprint = blueprints[n % len(blueprints)]  # 按顺序循环选择蓝图
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform))

            for response in self.client.apply_batch_sync(batch):
                if response.error:
                    continue
                else:
                    self.actors["non_agents"].append(response.actor_id)

        # 生成行人
        num_of_walkers = self.config["CARLA_CONFIG"]["NUM_OF_WALKERS"]
        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_locations = [self.world.get_random_location_from_navigation() for _ in range(num_of_walkers)]
        walker_locations = sorted(walker_locations, key=lambda x: str(x))  # 固定位置顺序

        batch = []
        for spawn_point in walker_locations:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            transform = carla.Transform(
                location=spawn_point,
                rotation=carla.Rotation(yaw=0.0)  # 添加默认旋转值
            )
            batch.append(carla.command.SpawnActor(walker_bp, transform))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                self.actors["walkers"].append(response.actor_id)
        print("spawn {} vehicles and {} walkers".format(len(self.actors["non_agents"]),
                                                        len(self.actors["walkers"])))
        self.world.tick()

    def set_actors_route(self):
        """
            设置actors自动运动
        """
        # 设置车辆Autopilot
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_synchronous_mode(True)
        vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # 设置行人随机运动
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for i in range(len(self.actors["walkers"])):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
                                                  self.actors["walkers"][i]))
        controllers_id = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        self.world.set_pedestrians_cross_factor(0.2)

        nav_points = [self.world.get_random_location_from_navigation() for _ in range(100)]
        nav_points = sorted(nav_points, key=lambda x: str(x))  # 固定导航点顺序
        for i, walker_id in enumerate(controllers_id):
            destination = nav_points[i % len(nav_points)]  # 按固定顺序选择目标
            self.world.get_actor(walker_id).go_to_location(destination)
            self.world.get_actor(walker_id).set_max_speed(1.4 + (i % 10)*0.2)  # 固定速度变化模式

    def spawn_agent(self):
        """
            生成agent（用于放置传感器的车辆与传感器）
        """
        spawn_points = sorted(self.world.get_map().get_spawn_points(), key=lambda x: str(x.location))  # 固定生成点顺序
        vehicle_bp = sorted(self.world.get_blueprint_library().filter(
            self.config["AGENT_CONFIG"]["BLUEPRINT"]), key=lambda bp: bp.id)[0]  # 固定选择第一个蓝图
        config_transform = self.config["AGENT_CONFIG"]["TRANSFORM"]
        carla_transform = config_transform_to_carla_transform(config_transform)
        
        # 检查生成位置是否空闲
        for transform in spawn_points:
            # 检查是否有车辆在该位置附近
            is_location_free = True
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.get_location().distance(transform.location) < 2.0:
                    is_location_free = False
                    break
            
            if is_location_free:
                try:
                    agent = self.world.spawn_actor(vehicle_bp, transform)
                    print("spawn agent success, transform: ", transform)
                    self.agent = agent
                    
                    # 保存代理的位姿
                    self.agent_transform = agent.get_transform()
                    
                    agent.set_autopilot(True, self.traffic_manager.get_port())
                    self.actors["agents"].append(agent)
                    break
                except RuntimeError as e:
                    logging.warning(f"Spawn failed at {transform.location}: {e}")
                    continue

        # 生成config中预设的传感器
        self.actors["sensors"][agent] = []
        for sensor_name, config in self.config["SENSOR_CONFIG"].items():
            sensor_bp = self.world.get_blueprint_library().find(config["BLUEPRINT"])
            for attr, val in config["ATTRIBUTE"].items():
                sensor_bp.set_attribute(attr, str(val))
            config_transform = config["TRANSFORM"]
            carla_transform = config_transform_to_carla_transform(config_transform)
            sensor_actor = self.world.spawn_actor(sensor_bp, carla_transform, attach_to=agent)
            
            # 打印传感器的名字
            print(f"Spawned sensor: {sensor_name}")
            
            self.actors["sensors"][agent].append(sensor_actor)
        self.world.tick()

    def set_spectator(self):
        """
            设置观察视角(与RGB相机一致)
        """
        spectator = self.world.get_spectator()

        # agent(放置传感器的车辆)位姿「相对世界坐标系」
        agent_transform_config = self.config["AGENT_CONFIG"]["TRANSFORM"]
        agent_transform = config_transform_to_carla_transform(agent_transform_config)

        # RGB相机位姿「相对agent坐标系」
        rgb_transform_config = self.config["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]
        rgb_transform = config_transform_to_carla_transform(rgb_transform_config)

        # spectator位姿「相对世界坐标系」
        spectator_location = agent_transform.location + rgb_transform.location
        spectator_rotation = carla.Rotation(yaw=agent_transform.rotation.yaw + rgb_transform.rotation.yaw,
                                            pitch=agent_transform.rotation.pitch + rgb_transform.rotation.pitch,
                                            roll=agent_transform.rotation.roll + rgb_transform.rotation.roll)
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
        
        
    def set_recover(self):
        """
            数据采集结束后，恢复默认设置
        """
        self.world.apply_settings(self.init_settings)
        self.traffic_manager.set_synchronous_mode(False)
        batch = []
        for actor_id in self.actors["non_agents"]:
            batch.append(carla.command.DestroyActor(actor_id))
        for walker_id in self.actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        for agent in self.actors["agents"]:
            for sensor in self.actors["sensors"][agent]:
                sensor.destroy()
            agent.destroy()
        self.client.apply_batch_sync(batch)

    def listen_sensor_data(self):
        """
            监听传感器信息
        """
        for agent, sensors in self.actors["sensors"].items():
            self.data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                self.data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

    def retrieve_data(self, q):
        """
            检查并获取传感器数据

            参数：
                q: CARLA原始数据

            返回：
                data：检查后的数据
        """
        while True:
            data = q.get()
            # 检查传感器数据与场景是否处于同一帧
            if data.frame == self.frame:
                return data

    def record_tick(self):
        """
            记录帧

            返回：
                data：CARLA传感器相关数据（原始数据，内参，外参等）
        """
        data = {"environment_objects": None, "actors": None, "agents_data": {}}
        self.frame = self.world.tick()

        data["environment_objects"] = self.world.get_environment_objects(carla.CityObjectLabel.Any)
        data["actors"] = self.world.get_actors()

        # 生成RGB图像的分辨率
        image_width = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]

        for agent, dataQue in self.data["sensor_data"].items():
            original_data = [self.retrieve_data(q) for q in dataQue]
            assert all(x.frame == self.frame for x in original_data)
            data["agents_data"][agent] = {}
            data["agents_data"][agent]["sensor_data"] = original_data
            
            # 设置传感器内参（仅相机有内参）
            data["agents_data"][agent]["intrinsic"] = set_camera_intrinsic(image_width, image_height)
            
            # 设置传感器外参
            data["agents_data"][agent]["extrinsic"] = np.mat(
                self.actors["sensors"][agent][0].get_transform().get_matrix())
            
            # 设置传感器的carla位姿
            data["agents_data"][agent]["transform"] = self.actors["sensors"][agent][0].get_transform()
            
            # 设置传感器的种类
            data["agents_data"][agent]["type"] = agent
            
            # 获取代理的速度
            velocity = agent.get_velocity()
            data["agents_data"][agent]["velocity"] = {
                "x": velocity.x,
                "y": velocity.y,
                "z": velocity.z
            }
            
            # 获取代理的加速度
            acceleration = agent.get_acceleration()
            data["agents_data"][agent]["acceleration"] = {
                "x": acceleration.x,
                "y": acceleration.y,
                "z": acceleration.z
            }
            

        # 根据预设距离对场景中的物体进行过滤
        data = object_filter_by_distance(data, self.config["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"])

        dataset = spawn_dataset(data)

        return dataset
    
    def get_agent_transform(self):
        """
            获取agent的位姿
        """
        self.agent_transform = self.agent.get_transform()
        return self.agent_transform

    def update_spectator(self):
        """
            更新观察者（spectator）的位姿，使其跟随代理。

            观察者将被设置在代理的上方，以便从上方观察代理的运动。

            返回：
                spectator: 更新后的观察者对象。
        """
        try:
            spectator = self.world.get_spectator()
            transform = self.get_agent_transform()
            bv_transform = carla.Transform(transform.location + carla.Location(z=40, x=0),
                                           carla.Rotation(yaw=0, pitch=-90))
            spectator.set_transform(bv_transform)
            print("set spectator success, transform: ", spectator.get_transform())
            return spectator
        except Exception as e:
            print(f"设置观察者位置和方向时出现错误：{e}")
