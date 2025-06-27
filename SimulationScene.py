import carla
import random
import logging
import queue
import numpy as np
from utils.utils import config_transform_to_carla_transform, set_camera_intrinsic, object_filter_by_distance
from utils.label import spawn_dataset
import sys
import os

# 添加Scenario Runner到系统路径
sys.path.append(os.environ["SCENARIO_RUNNER_ROOT"])

from scenario_runner import ScenarioRunner
import argparse
import time

class SimulationScene:
    def __init__(self, config):
        """
            初始化

            参数：
                config：预设配置
        """
        self.config = config
        self.args = None
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
        self.random_seed = 17  # 固定随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 新增：waypoint可视化相关
        self.debug = self.world.debug if self.world else None
        self.visualized_waypoints = []  # 存储已可视化的waypoint
        self.waypoint_lifetime = 60.0  # waypoint显示时间（秒）

    def set_map(self):
        """
            设置场景地图
        """
        self.world = self.client.get_world()
        self.debug = self.world.debug  # 初始化debug接口

    def visualize_waypoints_around_vehicle(self, vehicle=None, distance=50.0, interval=2.0):
        """
        可视化车辆周围的waypoints
        
        参数:
            vehicle: 车辆对象，如果为None则使用self.agent
            distance: 可视化范围（米）
            interval: waypoint间隔（米）
        """
        if vehicle is None:
            vehicle = self.agent
            
        if vehicle is None or self.debug is None:
            return
            
        # 获取车辆当前位置的waypoint
        current_location = vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(current_location)
        
        if current_waypoint is None:
            return
            
        # 生成前方waypoints
        waypoints = [current_waypoint]
        wp = current_waypoint
        
        # 向前生成waypoints
        forward_distance = 0
        while forward_distance < distance:
            next_wps = wp.next(interval)
            if next_wps:
                wp = next_wps[0]
                waypoints.append(wp)
                forward_distance += interval
            else:
                break
                
        # 向后生成waypoints
        wp = current_waypoint
        backward_distance = 0
        while backward_distance < distance / 2:  # 向后显示一半距离
            prev_wps = wp.previous(interval)
            if prev_wps:
                wp = prev_wps[0]
                waypoints.insert(0, wp)  # 插入到开头
                backward_distance += interval
            else:
                break
        
        # 可视化waypoints
        self.draw_waypoints_with_numbers(waypoints)

    def draw_waypoints_with_numbers(self, waypoints, color=None):
        """
        绘制waypoints并显示编号
        
        参数:
            waypoints: waypoint列表
            color: 颜色，如果为None则使用默认颜色
        """
        if self.debug is None:
            return
            
        if color is None:
            color = carla.Color(255, 0, 0)  # 红色
            
        for i, waypoint in enumerate(waypoints):
            location = waypoint.transform.location
            
            # 绘制waypoint点
            self.debug.draw_point(
                location + carla.Location(z=0.5),  # 稍微抬高一点
                size=0.1,
                color=color,
                life_time=self.waypoint_lifetime
            )
            
            # 绘制编号
            self.debug.draw_string(
                location + carla.Location(z=1.5),  # 在点上方显示文字
                str(i),
                draw_shadow=True,
                color=carla.Color(255, 255, 255),  # 白色文字
                life_time=self.waypoint_lifetime
            )
            
            # 绘制方向箭头
            if i < len(waypoints) - 1:
                next_location = waypoints[i + 1].transform.location
                self.debug.draw_arrow(
                    location + carla.Location(z=0.3),
                    next_location + carla.Location(z=0.3),
                    thickness=0.05,
                    arrow_size=0.2,
                    color=carla.Color(0, 255, 0),  # 绿色箭头
                    life_time=self.waypoint_lifetime
                )

    def visualize_lane_change_path(self, start_waypoint, target_lateral_offset, num_steps=20):
        """
        可视化变道路径
        
        参数:
            start_waypoint: 起始waypoint
            target_lateral_offset: 目标横向偏移（正值向右，负值向左）
            num_steps: 变道步骤数
        """
        if self.debug is None:
            return
            
        waypoints = []
        current_wp = start_waypoint
        
        # 生成变道路径
        for i in range(num_steps):
            # 计算当前的横向偏移
            progress = i / (num_steps - 1)
            current_offset = target_lateral_offset * self._smooth_step(progress)
            
            # 计算偏移后的位置
            right_vector = current_wp.transform.get_right_vector()
            offset_location = current_wp.transform.location + carla.Location(
                x=current_offset * right_vector.x,
                y=current_offset * right_vector.y,
                z=0.5
            )
            
            # 创建虚拟waypoint用于可视化
            class VirtualWaypoint:
                def __init__(self, location):
                    self.transform = carla.Transform(location)
                    
            virtual_wp = VirtualWaypoint(offset_location)
            waypoints.append(virtual_wp)
            
            # 移动到下一个waypoint
            next_wps = current_wp.next(2.0)
            if next_wps:
                current_wp = next_wps[0]
            else:
                break
                
        # 使用蓝色绘制变道路径
        self.draw_waypoints_with_numbers(waypoints, carla.Color(0, 0, 255))

    def _smooth_step(self, t):
        """
        平滑步进函数，创建S形曲线
        """
        import math
        return 0.5 * (1 + math.sin(math.pi * (t - 0.5)))

    def visualize_vehicle_path_history(self, vehicle=None, max_points=50):
        """
        可视化车辆行驶轨迹
        
        参数:
            vehicle: 车辆对象
            max_points: 最大轨迹点数
        """
        if vehicle is None:
            vehicle = self.agent
            
        if vehicle is None or self.debug is None:
            return
            
        # 如果没有历史轨迹列表，创建一个
        if not hasattr(self, 'vehicle_path_history'):
            self.vehicle_path_history = []
            
        # 添加当前位置到轨迹
        current_location = vehicle.get_location()
        self.vehicle_path_history.append(current_location)
        
        # 限制轨迹长度
        if len(self.vehicle_path_history) > max_points:
            self.vehicle_path_history.pop(0)
            
        # 绘制轨迹
        for i, location in enumerate(self.vehicle_path_history):
            # 颜色从暗到亮，表示时间先后
            intensity = int(255 * (i + 1) / len(self.vehicle_path_history))
            color = carla.Color(intensity, intensity, 0)  # 黄色渐变
            
            self.debug.draw_point(
                location + carla.Location(z=0.2),
                size=0.05,
                color=color,
                life_time=1.0
            )

    def clear_debug_shapes(self):
        """
        清除所有debug绘制的形状
        """
        if self.debug:
            # CARLA没有直接的清除方法，只能等待life_time过期
            # 或者重新设置较短的life_time
            pass

    def update_waypoint_visualization(self):
        """
        更新waypoint可视化（在每帧调用）
        """
        if self.agent:
            # 可视化车辆周围的waypoints
            self.visualize_waypoints_around_vehicle(self.agent, distance=30.0, interval=3.0)
            
            # 可视化车辆轨迹
            self.visualize_vehicle_path_history(self.agent)

    # 在原有代码基础上修改record_tick方法
    def record_tick(self):
        """
            记录帧

            返回：
                data：CARLA传感器相关数据（原始数据，内参，外参等）
        """
        data = {"environment_objects": None, "actors": None, "agents_data": {}}
        self.frame = self.world.tick()
        
        # 新增：更新waypoint可视化
        self.update_waypoint_visualization()
        
        # 新增ego状态记录
        if self.agent:
            transform = self.agent.get_transform()
            velocity = self.agent.get_velocity()
            acceleration = self.agent.get_acceleration()
            transform_matrix = self.agent.get_transform().get_matrix()
            data["egostate"] = {
                "location": {
                    "x": transform.location.x,
                    "y": -transform.location.y,
                    "z": transform.location.z
                },
                "rotation": {
                    "roll": -transform.rotation.roll,
                    "pitch": transform.rotation.pitch,
                    "yaw": -transform.rotation.yaw
                },
                "velocity": {
                    "x": velocity.x,
                    "y": -velocity.y,
                    "z": velocity.z
                },
                "acceleration": {
                    "x": acceleration.x,
                    "y": -acceleration.y,
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

    # 新增：手动触发可视化的方法
    def show_waypoints_ahead(self, distance=50.0):
        """
        手动显示前方waypoints
        """
        if self.agent:
            self.visualize_waypoints_around_vehicle(self.agent, distance=distance)

    def show_lane_change_preview(self, lateral_offset=-3.5):
        """
        预览变道路径
        
        参数:
            lateral_offset: 横向偏移量（负值向左，正值向右）
        """
        if self.agent:
            current_location = self.agent.get_location()
            current_waypoint = self.world.get_map().get_waypoint(current_location)
            if current_waypoint:
                self.visualize_lane_change_path(current_waypoint, lateral_offset)

    # 保持原有的其他方法不变...
    def set_weather(self):
        """
            设置场景天气
        """
        weather = carla.WeatherParameters.ClearNoon
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
            收集ScenarioRunner生成的actors(车辆与行人)
        """
        # 收集现有车辆（排除ego车辆）
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.attributes.get('role_name', '') != 'hero':  # 排除ego车辆
                self.actors["non_agents"].append(vehicle.id)

        # 收集现有行人
        walkers = self.world.get_actors().filter('walker.pedestrian.*')
        for walker in walkers:
            self.actors["walkers"].append(walker.id)

        print(f"已找到 {len(self.actors['non_agents'])} 辆NPC车辆和 {len(self.actors['walkers'])} 个行人")

    def set_actors_route(self):
        """
            设置actors自动运动
        """
        pass

    def spawn_agent(self):
        """
            生成agent（通过ScenarioRunner自动寻找ego车辆）
        """
        
        # 删除原有生成逻辑，改为自动寻找
        start_time = time.time()
        while time.time() - start_time < 10.0:  # 最多等待10秒
            time.sleep(0.5)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            
            for vehicle in possible_vehicles:
                if vehicle.attributes.get('role_name', '') == 'hero':
                    self.agent = vehicle
                    print(f"找到ego车辆: {self.agent.type_id} (ID: {self.agent.id})")
                    self.actors["agents"].append(self.agent)
                    self._spawn_sensors(self.agent)  # 挂载传感器
                    # 设置自动驾驶模式
                    self.agent.set_autopilot(True)
                    self.world.tick()
                    return

            if self.agent:
                break

        if not self.agent:
            raise RuntimeError("未能在10秒内找到ego车辆（role_name='hero'）")

    def _spawn_sensors(self, agent):
        """
            为指定agent生成传感器
        
        参数：
            agent: 需要安装传感器的车辆对象
        """
        # 初始化传感器字典
        self.actors["sensors"][agent] = []
        
        # 遍历配置生成传感器
        for sensor_name, config in self.config["SENSOR_CONFIG"].items():
            sensor_bp = self.world.get_blueprint_library().find(config["BLUEPRINT"])
            
            # 设置传感器属性
            for attr, val in config["ATTRIBUTE"].items():
                sensor_bp.set_attribute(attr, str(val))
                
            # 转换坐标系
            config_transform = config["TRANSFORM"]
            carla_transform = config_transform_to_carla_transform(config_transform)
            
            # 生成并挂载传感器
            sensor_actor = self.world.spawn_actor(
                sensor_bp, 
                carla_transform, 
                attach_to=agent
            )
            self.actors["sensors"][agent].append(sensor_actor)
            
            # 调试信息
            logging.debug(f"已挂载传感器: {sensor_name} -> {sensor_actor.type_id}")

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
            bv_transform = carla.Transform(transform.location + carla.Location(z=25, x=0, y=0),
                                        carla.Rotation(roll=0, yaw=0, pitch=-90))
            spectator.set_transform(bv_transform)
            return spectator
        except Exception as e:
            print(f"设置观察者位置和方向时出现错误：{e}")