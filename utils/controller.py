"""
轨迹控制器模块

包含基于CARLA PID控制器的轨迹跟踪控制器
"""

import sys
import os
import glob
import carla

# 添加CARLA Python API路径
CARLA_ROOT = os.environ.get('CARLA_ROOT', '/opt/carla-simulator')
sys.path.append(f'{CARLA_ROOT}/PythonAPI/carla')
sys.path.append(f'{CARLA_ROOT}/PythonAPI/carla/agents')

try:
    sys.path.append(glob.glob(f'{CARLA_ROOT}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle

class CarlaTrajectoryController:
    """
    基于CARLA PID控制器的轨迹跟踪控制器
    """
    
    def __init__(self, vehicle, world, target_speed=30.0):
        """
        初始化PID控制器
        
        Args:
            vehicle: CARLA车辆对象
            world: CARLA世界对象
            target_speed: 目标速度 (km/h)
        """
        self.vehicle = vehicle
        self.world = world
        self.target_speed = target_speed
        
        # PID参数配置
        args_lateral_dict = {
            'K_P': 1.95, 
            'K_D': 0.2, 
            'K_I': 0.07, 
            'dt': 1.0 / 10.0
        }
        
        args_long_dict = {
            'K_P': 1, 
            'K_D': 0.0, 
            'K_I': 0.75, 
            'dt': 1.0 / 10.0
        }
        
        # 创建PID控制器
        self.pid_controller = VehiclePIDController(
            vehicle, 
            args_lateral=args_lateral_dict, 
            args_longitudinal=args_long_dict
        )
        
        # 轨迹相关
        self.waypoints = []
        self.current_waypoint_index = 0
        self.reached_goal = False
        self.waypoint_threshold = 2.0  # 到达waypoint的距离阈值
        
    def set_waypoints(self, transform_list):
        """
        设置要跟随的轨迹点
        
        Args:
            transform_list: carla.Transform列表
        """
        # 将Transform列表转换为包含transform属性的对象
        class WaypointWrapper:
            def __init__(self, transform):
                self.transform = transform
                
        self.waypoints = [WaypointWrapper(transform) for transform in transform_list]
        self.current_waypoint_index = 0
        self.reached_goal = False
        
    def set_target_speed(self, speed_kmh):
        """设置目标速度 (km/h)"""
        self.target_speed = speed_kmh
        
    def get_remaining_waypoints(self):
        """获取剩余的waypoints数量"""
        if not self.waypoints:
            return 0
        return len(self.waypoints) - self.current_waypoint_index
        
    def run_step(self):
        """执行一步控制"""
        if not self.vehicle or not self.vehicle.is_alive:
            return
            
        if not self.waypoints or self.reached_goal:
            # 没有轨迹或已到达目标，停车
            control = carla.VehicleControl()
            control.brake = 1.0
            control.throttle = 0.0
            self.vehicle.apply_control(control)
            return
            
        if self.current_waypoint_index >= len(self.waypoints):
            self.reached_goal = True
            return
            
        # 获取当前目标waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # 检查是否到达当前waypoint
        ego_transform = self.vehicle.get_transform()
        distance_to_waypoint = distance_vehicle(current_waypoint, ego_transform)
        
        # 如果接近当前waypoint，切换到下一个
        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
            
            # 检查是否到达最后一个waypoint
            if self.current_waypoint_index >= len(self.waypoints):
                self.reached_goal = True
                # 最后一个waypoint，减速停车
                control = self.pid_controller.run_step(0, current_waypoint)
            else:
                # 切换到下一个waypoint
                current_waypoint = self.waypoints[self.current_waypoint_index]
                control = self.pid_controller.run_step(self.target_speed, current_waypoint)
        else:
            # 继续朝当前waypoint行驶
            control = self.pid_controller.run_step(self.target_speed, current_waypoint)
            
        # 应用控制命令
        self.vehicle.apply_control(control)
        
        # 可视化当前目标点（已禁用）
        # if hasattr(self, 'world') and self.world:
        #     ego_loc = self.vehicle.get_location()
        #     target_loc = current_waypoint.transform.location
            
        #     # 绘制ego位置和目标点
        #     self.world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=0.1, size=0.1)
        #     self.world.debug.draw_point(target_loc, color=carla.Color(g=255), life_time=0.1, size=0.1)
            
    def is_reached_goal(self):
        """检查是否到达目标"""
        return self.reached_goal
        
    def reset(self):
        """重置控制器"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.reached_goal = False
        if self.vehicle and self.vehicle.is_alive:
            control = carla.VehicleControl()
            control.brake = 1.0
            control.throttle = 0.0
            self.vehicle.apply_control(control)