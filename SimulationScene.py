import carla
import random
import logging
import queue
import numpy as np
from utils.utils import config_transform_to_carla_transform, set_camera_intrinsic, object_filter_by_distance
from utils.label import spawn_dataset
# from scenario_runner import ScenarioRunner

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
        # 47 or 17
        self.random_seed = 47  # 固定随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def set_map(self):
        """
            设置场景地图
        """
        self.world = self.client.load_world('Town10HD_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Foliage)  # 移除植被层优化性能


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
        # 固定时间步长 (0.01s)
        settings.fixed_delta_seconds = 0.01
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
        
        # 生成危险行人（精确坐标）
        # 生成多个危险行人
        self.jaywalkers = []  # 存储所有危险行人
        jaywalker_configs = [
            {
                "location": carla.Location(x=-3.97, y=36.10, z=0.80),
                "yaw": -89.84,
                "target_offset": {"x": 0, "y": -15},
                "speed": 1.8 
            },
            {
                "location": carla.Location(x=-2.17, y=37.10, z=0.80),
                "yaw": -89.84,
                "target_offset": {"x": 0, "y": -15},
                "speed": 2.0
            },
            {
                "location": carla.Location(x=-3.07, y=22.70, z=0.80),
                "yaw": 90.16,
                "target_offset": {"x": 0, "y": 15},
                "speed": 1.5 
            },
            {
                "location": carla.Location(x=3.97, y=35.10, z=0.80),
                "yaw": -45.0,
                "target_offset": {"x": -5, "y": -10},  # 斜向移动
                "speed": 3.0 
            }
        ]

        for cfg in jaywalker_configs:
            try:
                # 生成行人
                transform = carla.Transform(
                    location=cfg["location"],
                    rotation=carla.Rotation(yaw=cfg["yaw"])
                )
                
                walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
                jaywalker = self.world.spawn_actor(walker_bp, transform)
                self.actors["walkers"].append(jaywalker)
                self.jaywalkers.append({
                    "actor": jaywalker,
                    "start_pos": cfg["location"],
                    "target_offset": cfg["target_offset"]
                })

                # # 可视化设置
                # self.world.debug.draw_point(
                #     transform.location, 
                #     size=0.3, 
                #     color=carla.Color(255, 0, 0),
                #     life_time=600.0
                # )
                # self.world.debug.draw_arrow(
                #     transform.location,
                #     transform.location + carla.Location(
                #         x=2 * np.cos(np.deg2rad(transform.rotation.yaw)),
                #         y=2 * np.sin(np.deg2rad(transform.rotation.yaw)),
                #         z=0
                #     ),
                #     thickness=0.1,
                #     arrow_size=0.3,
                #     color=carla.Color(0, 255, 0),
                #     life_time=600.0
                # )

            except Exception as e:
                logging.error(f"危险行人生成失败: {str(e)}")

        self.world.tick()

    def set_actors_route(self):
        """
            设置actors自动运动
        """
        # 设置车辆Autopilot
        self.traffic_manager.set_global_distance_to_leading_vehicle(0.5)
        self.traffic_manager.set_synchronous_mode(True)
        # vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        # for vehicle in vehicle_actors:
        #     vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # # 设置行人固定路线运动
        # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        # batch = []
        # for i in range(len(self.actors["walkers"])):
        #     batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
        #                                           self.actors["walkers"][i]))
        # controllers_id = []
        # for response in self.client.apply_batch_sync(batch, True):
        #     if response.error:
        #         logging.warning(f"Failed to spawn walker controller: {response.error}")
        #         continue
        #     else:
        #         controllers_id.append(response.actor_id)
        
        # # 获取所有导航点
        # all_spawn_points = self.world.get_map().get_spawn_points()
        
        # # 确保有足够的导航点
        # if len(all_spawn_points) < len(controllers_id):
        #     raise ValueError(f"Not enough spawn points. Need {len(controllers_id)}, have {len(all_spawn_points)}")
        
        # for i, walker_id in enumerate(controllers_id):
        #     try:
        #         controller = self.world.get_actor(walker_id)
        #         # 启动控制器
        #         controller.start()
                
        #         # 使用固定索引获取导航点
        #         destination = all_spawn_points[i % len(all_spawn_points)].location
                
        #         # 设置目标位置
        #         controller.go_to_location(destination)
                
        #         # 设置合理速度 (1.0 - 2.0 m/s)
        #         speed = 1.5  # 固定速度
        #         controller.set_max_speed(max(1.0, min(speed, 2.0)))
                
        #     except Exception as e:
        #         logging.warning(f"Failed to configure walker {walker_id}: {str(e)}")
        #         continue
        
         # 新增危险行人控制（与data_collection一致）
        # 控制所有危险行人
        for jaywalker_info in self.jaywalkers:
            jaywalker = jaywalker_info["actor"]
            if not jaywalker.is_alive:
                continue

            try:
                # 计算目标位置
                target_location = carla.Location(
                x=jaywalker_info["start_pos"].x + jaywalker_info["target_offset"].get("x", 0),
                y=jaywalker_info["start_pos"].y + jaywalker_info["target_offset"].get("y", 0),
                z=jaywalker_info["start_pos"].z
            )

                # 创建控制指令
                walker_control = carla.WalkerControl()
                direction = target_location - jaywalker.get_location()
                walker_control.direction = direction.make_unit_vector()
                walker_control.speed = jaywalker_info.get("speed", 1.8) # 如果没有speed使用默认值1.8
                walker_control.jump = False
                
                # 应用控制
                jaywalker.apply_control(walker_control)

                # # 动态可视化路径
                # self.world.debug.draw_arrow(
                #     jaywalker.get_location(),
                #     target_location,
                #     thickness=0.1,
                #     arrow_size=0.2,
                #     color=carla.Color(random.randint(0,255), random.randint(0,255), 0),
                #     life_time=1.0/self.config.get("FPS", 20)
                # )

            except Exception as e:
                logging.error(f"行人控制失败: {str(e)}")


        if self.agent:
            # traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.ignore_lights_percentage(self.agent, 100)  # 忽略所有交通灯
            self.traffic_manager.auto_lane_change(self.agent, False)  # 禁止自动变道
            self.traffic_manager.vehicle_percentage_speed_difference(self.agent, 30)  # 限速30%


                

    def spawn_agent(self):
        """
            生成agent（用于放置传感器的车辆与传感器）
        """
        # 创建精确的生成点
        ego_transform = carla.Transform(
            location=carla.Location(
                x=-7.97,  # 精确X坐标
                y=28.10,   # 精确Y坐标
                z=0.80     # 精确Z高度
            ),
            rotation=carla.Rotation(
                yaw=0.16   # 精确偏航角
            )
        )

        # 保持原有生成逻辑
        vehicle_bp = sorted(self.world.get_blueprint_library().filter(
            self.config["AGENT_CONFIG"]["BLUEPRINT"]), key=lambda bp: bp.id)[0]
        
        # 尝试生成
        try:
            agent = self.world.spawn_actor(vehicle_bp, ego_transform)
            print(f"主车生成成功 坐标: {ego_transform.location}")
            self.agent = agent
            # agent.set_autopilot(True, self.traffic_manager.get_port())
            agent.set_autopilot(False)
            self.actors["agents"].append(agent)
        except RuntimeError as e:
            logging.error(f"主车生成失败: {str(e)}")
            raise

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
            # print(f"Spawned sensor: {sensor_name}")
            
            self.actors["sensors"][agent].append(sensor_actor)
        self.world.tick()

    def set_spectator(self):
        """
            设置观察视角(与RGB相机一致)
        """
        spectator = self.world.get_spectator()
        transform = self.get_agent_transform()
        bv_transform = carla.Transform(transform.location + carla.Location(z=40, x=0, y=0),
                                        carla.Rotation(roll=0, yaw=0, pitch=-90))
        spectator.set_transform(bv_transform)
        
        
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
        
        
        # 新增ego状态记录
        if self.agent:
            transform = self.agent.get_transform()
            velocity = self.agent.get_velocity()
            acceleration = self.agent.get_acceleration()
            transform_matrix = self.agent.get_transform().get_matrix()
            data["egostate"] = {
                "location": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z
                },
                "rotation": {
                    "roll": transform.rotation.roll,
                    "pitch": transform.rotation.pitch,
                    "yaw": transform.rotation.yaw
                },
                "velocity": {
                    "x": velocity.x,
                    "y": velocity.y,
                    "z": velocity.z
                },
                "acceleration": {
                    "x": acceleration.x,
                    "y": acceleration.y,
                    "z": acceleration.z
                },
                "extent": {  # 车辆包围盒尺寸
                    "x": self.agent.bounding_box.extent.x,
                    "y": self.agent.bounding_box.extent.y,
                    "z": self.agent.bounding_box.extent.z
                },
                "matrix": np.array(transform_matrix)
            }
        data["environment_objects"] = self.world.get_environment_objects(carla.CityObjectLabel.Any)

        data["actors"] = self.world.get_actors()

        # 生成RGB图像的分辨率
        image_width = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]

        for agent, dataQue in self.data["sensor_data"].items():
            original_data = [self.retrieve_data(q) for q in dataQue]
            print("original_data retrive success")
            assert all(x.frame == self.frame for x in original_data)
            data["agents_data"][agent] = {}
            data["agents_data"][agent]["sensor_data"] = original_data
            
            # 设置传感器内参（仅相机有内参）
            data["agents_data"][agent]["intrinsic"] = set_camera_intrinsic(image_width, image_height)
            
            # 设置传感器的carla位姿
            # data["agents_data"][agent]["transform"] = self.actors["sensors"][agent][0].get_transform()
            data["agents_data"][agent]["transform"] = [
                np.mat(sensor.get_transform().get_matrix())
                for sensor in self.actors["sensors"][agent]
            ]
            
            # 获取ego车辆的逆变换矩阵
            ego_transform = self.agent.get_transform()
            ego_inv_matrix = np.mat(ego_transform.get_inverse_matrix())
            
            # 计算传感器到ego的相对外参
            data["agents_data"][agent]["extrinsic"] = [
                np.mat(np.where(np.abs(sensor.get_transform().get_matrix()) < 0.0001, 0, sensor.get_transform().get_matrix())) * ego_inv_matrix
                for sensor in self.actors["sensors"][agent]
            ]
                     
            
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
            
        print("Get data success")
        # 根据预设距离对场景中的物体进行过滤
        data = object_filter_by_distance(data, self.config["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"])
        print("Filter data success")
        dataset = spawn_dataset(data)
        print("Spawn dataset success")

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
            bv_transform = carla.Transform(transform.location + carla.Location(z=40, x=0, y=0),
                                        carla.Rotation(roll=0, yaw=0, pitch=-90))
            spectator.set_transform(bv_transform)
            return spectator
        except Exception as e:
            print(f"设置观察者位置和方向时出现错误：{e}")
