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
    带调试功能的轨迹跟踪控制器
    """
    
    def __init__(self, vehicle, world, target_speed=30.0):
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
        
        self.pid_controller = VehiclePIDController(
            vehicle, 
            args_lateral=args_lateral_dict, 
            args_longitudinal=args_long_dict
        )
        
        # 轨迹相关
        self.waypoints = []
        self.current_waypoint_index = 0
        self.reached_goal = False
        self.waypoint_threshold = 3.0  # 增加阈值到3米
        
        # 调试相关
        self.debug_counter = 0
        
    def set_waypoints(self, transform_list):
        """设置要跟随的轨迹点并进行优化"""
        if not transform_list:
            return
            
        # 过滤掉过于接近的waypoints
        filtered_waypoints = self._filter_close_waypoints(transform_list, min_distance=2.0)
        
        # 将Transform列表转换为包含transform属性的对象
        class WaypointWrapper:
            def __init__(self, transform):
                self.transform = transform
                
        self.waypoints = [WaypointWrapper(transform) for transform in filtered_waypoints]
        self.current_waypoint_index = 0
        self.reached_goal = False
        
        print(f"原始waypoints: {len(transform_list)}, 过滤后: {len(self.waypoints)}")
        
    def _filter_close_waypoints(self, transform_list, min_distance=2.0):
        """过滤掉距离太近的waypoints"""
        if not transform_list:
            return []
            
        filtered = [transform_list[0]]  # 保留第一个点
        
        for transform in transform_list[1:]:
            last_transform = filtered[-1]
            distance = last_transform.location.distance(transform.location)
            
            if distance >= min_distance:
                filtered.append(transform)
            
        return filtered
        
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
            
        self.debug_counter += 1
        
        if not self.waypoints:
            print("没有waypoints，停车")
            self._apply_stop_control()
            return
            
        if self.reached_goal:
            if self.debug_counter % 30 == 0:  # 每30帧打印一次
                print("已到达目标，保持停车状态")
            self._apply_stop_control()
            return
            
        if self.current_waypoint_index >= len(self.waypoints):
            print("超出waypoints范围，设置为已到达目标")
            self.reached_goal = True
            self._apply_stop_control()
            return
            
        # 获取当前目标waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        ego_transform = self.vehicle.get_transform()
        distance_to_waypoint = distance_vehicle(current_waypoint, ego_transform)
        
        # 调试信息
        if self.debug_counter % 30 == 0:
            remaining = len(self.waypoints) - self.current_waypoint_index
            current_speed = self.vehicle.get_velocity()
            speed_kmh = (current_speed.x**2 + current_speed.y**2)**0.5 * 3.6
            print(f"Waypoint {self.current_waypoint_index}/{len(self.waypoints)-1}, "
                  f"距离: {distance_to_waypoint:.2f}m, "
                  f"剩余: {remaining}, "
                  f"速度: {speed_kmh:.1f}km/h")
        
        # 检查是否接近最后几个waypoints
        remaining_waypoints = len(self.waypoints) - self.current_waypoint_index
        
        if remaining_waypoints <= 2:  # 最后2个waypoint
            if distance_to_waypoint < self.waypoint_threshold:
                print(f"到达最后waypoint {self.current_waypoint_index}, 停车")
                self.reached_goal = True
                self._apply_stop_control()
                return
            else:
                # 最后waypoint，减速接近
                reduced_speed = min(self.target_speed * 0.3, 10.0)  # 减速到30%或10km/h
                control = self.pid_controller.run_step(reduced_speed, current_waypoint)
                self.vehicle.apply_control(control)
                return
        
        # 正常waypoint切换逻辑
        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
            print(f"切换到waypoint {self.current_waypoint_index}")
            
            # 检查是否到达最后一个waypoint
            if self.current_waypoint_index >= len(self.waypoints):
                print("到达最后一个waypoint，停车")
                self.reached_goal = True
                self._apply_stop_control()
                return
            else:
                # 切换到下一个waypoint
                current_waypoint = self.waypoints[self.current_waypoint_index]
                control = self.pid_controller.run_step(self.target_speed, current_waypoint)
        else:
            # 继续朝当前waypoint行驶
            control = self.pid_controller.run_step(self.target_speed, current_waypoint)
            
        # 应用控制命令
        self.vehicle.apply_control(control)
        
    def _apply_stop_control(self):
        """应用停车控制"""
        control = carla.VehicleControl()
        control.brake = 1.0
        control.throttle = 0.0
        control.steer = 0.0
        control.hand_brake = True
        self.vehicle.apply_control(control)
        
    def is_reached_goal(self):
        """检查是否到达目标"""
        return self.reached_goal
        
    def reset(self):
        """重置控制器"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.reached_goal = False
        self.debug_counter = 0
        if self.vehicle and self.vehicle.is_alive:
            self._apply_stop_control()