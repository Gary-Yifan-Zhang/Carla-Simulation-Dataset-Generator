# 使用PID控制器的变道轨迹跟踪主程序

from utils.utils import yaml_to_config
from SimulationScene import SimulationScene
from DatasetSave import DatasetSave
import time
import argparse
from utils.mask import process_all_masks
from utils.visual import images_to_video
from utils.pointcloud import batch_merge
from utils.controller import CarlaTrajectoryController  # 从utils.controller导入
import shutil
import carla
import math

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="仿真数据采集程序")
    parser.add_argument('--no-save', action='store_true', 
                        help="跳过数据保存步骤")
    parser.add_argument('--scenario-name', type=str, default="UnknownScenario",
                        help="当前运行的场景名称")
    parser.add_argument('--use-trajectory-control', action='store_true',
                        help="使用轨迹跟踪控制ego车辆")
    parser.add_argument('--lane-change-direction', type=str, choices=['left', 'right'], 
                        default='left', help="变道方向 (left/right)")
    args = parser.parse_args()
    
    program_start_time = time.time()

    # 加载配置文件
    config = yaml_to_config("configs.yaml")
    
    # 初始化仿真场景
    scene = SimulationScene(config)
    
    if not args.no_save:
        # 初始化保存设置
        dataset_save = DatasetSave(config, args.scenario_name)
        shutil.copy('calib.txt', dataset_save.OUTPUT_FOLDER)
        
    # 用于存储轨迹控制器
    ego_controller = None
    # 变道状态管理
    lane_change_state = "preparing"
    lane_change_stage = 0
    
    try:
        # 设置场景地图
        scene.set_map()
        # 设置场景天气
        scene.set_weather()
        # 开启同步模式
        scene.set_synchrony()
        # 生成agent（用于放置传感器的车辆与传感器）
        scene.spawn_agent()
        # 在场景中生成actors(车辆与行人)
        scene.spawn_actors()
        # 设置actors自动运动
        scene.set_actors_route()
        # 设置观察视角(与RGB相机一致)
        scene.set_spectator()

        # 获取变道轨迹数据
        print("获取变道轨迹数据...")
        
        lateral_offset = -3.6 if args.lane_change_direction == 'left' else 3.6
        print(f"设置变道方向: {'向左' if args.lane_change_direction == 'left' else '向右'}")
        
        lane_change_trajectories = scene.get_lane_change_trajectory(lateral_offset=lateral_offset)
        
        if not lane_change_trajectories or not any(lane_change_trajectories.values()):
            print("无法获取变道轨迹，回退到自动驾驶模式")
            args.use_trajectory_control = False
        
        # 选择要跟踪的轨迹
        if args.use_trajectory_control and lane_change_trajectories:
            print("=== 变道轨迹信息 ===")
            print(f"原始车道轨迹点数: {len(lane_change_trajectories['original'])}")
            print(f"变道过程轨迹点数: {len(lane_change_trajectories['lane_change'])}")
            print(f"变道后轨迹点数: {len(lane_change_trajectories['after_change'])}")
            print("==================")
            
            try:
                # 使用从utils.controller导入的PID控制器
                ego_controller = CarlaTrajectoryController(
                    scene.agent, 
                    scene.world, 
                    target_speed=40.0  # 40 km/h
                )
                print("CarlaTrajectoryController 创建成功")
                
                # 开始时先跟随原始车道轨迹
                if lane_change_trajectories['original']:
                    initial_waypoints = lane_change_trajectories['original']
                    ego_controller.set_waypoints(initial_waypoints)
                    print(f"设置初始轨迹（原始车道），包含 {len(initial_waypoints)} 个waypoint")
                    lane_change_stage = 0
                    lane_change_state = "preparing"
                else:
                    print("原始车道轨迹为空，无法开始变道")
                    ego_controller = None
                
                if ego_controller:
                    print(f"设置目标速度: {ego_controller.target_speed} km/h")
                    # 关闭自动驾驶，使用自定义控制器
                    scene.agent.set_autopilot(False)
                    
            except Exception as e:
                print(f"创建CarlaTrajectoryController失败: {e}")
                print("回退到自动驾驶模式")
                scene.agent.set_autopilot(True)
                ego_controller = None
        
        if not ego_controller:
            print("使用默认自动驾驶模式")
            scene.agent.set_autopilot(True)

        if not args.no_save: 
            # 监听传感器信息
            scene.listen_sensor_data()

        # 帧数
        frame = 0
        # 记录步长
        step = config["SAVE_CONFIG"]["STEP"]
        counter = 0
        # 获取最大记录次数配置（带默认值）
        max_record = config["SAVE_CONFIG"].get("MAX_RECORD_COUNT", float('inf'))

        # 新增初始化等待阶段
        print("初始化完成，开始预运行...")
        INIT_WAIT_FRAMES = 50
        for _ in range(INIT_WAIT_FRAMES):
            scene.update_spectator()
            
            if ego_controller:
                try:
                    ego_controller.run_step()
                except Exception as e:
                    print(f"控制器运行错误: {e}")
                    print("回退到自动驾驶模式")
                    ego_controller.reset()
                    ego_controller = None
                    scene.agent.set_autopilot(True)
                
            scene.world.tick()
            print(f"预运行进度: {_+1}/{INIT_WAIT_FRAMES}", end='\r')
        print("\n预运行完成，开始主循环")

        # 变道控制计数器
        trajectory_check_counter = 0
        TRAJECTORY_CHECK_INTERVAL = 30

        while True:
            # 变道轨迹跟踪控制逻辑
            if ego_controller and scene.agent and scene.agent.is_alive:
                try:
                    # 定期检查变道进度
                    if trajectory_check_counter % TRAJECTORY_CHECK_INTERVAL == 0:
                        remaining = ego_controller.get_remaining_waypoints()
                        
                        # 变道状态机
                        if lane_change_stage == 0 and remaining < 5:
                            # 切换到变道轨迹
                            if lane_change_trajectories['lane_change']:
                                lane_change_waypoints = lane_change_trajectories['lane_change']
                                ego_controller.set_waypoints(lane_change_waypoints)
                                lane_change_stage = 1
                                lane_change_state = "changing"
                                print(f"=== 开始变道 ===")
                                print(f"切换到变道轨迹，包含 {len(lane_change_waypoints)} 个waypoint")
                                ego_controller.set_target_speed(15.0)  # 变道时降速
                                
                        elif lane_change_stage == 1 and remaining < 3:
                            # 切换到变道后轨迹
                            if lane_change_trajectories['after_change']:
                                after_change_waypoints = lane_change_trajectories['after_change']
                                ego_controller.set_waypoints(after_change_waypoints)
                                lane_change_stage = 2
                                lane_change_state = "completed"
                                print(f"=== 变道完成 ===")
                                print(f"切换到变道后轨迹，包含 {len(after_change_waypoints)} 个waypoint")
                                ego_controller.set_target_speed(20.0)  # 恢复正常速度
                                
                        # elif lane_change_stage == 2 and remaining < 5:
                        #     # 重新获取新的变道轨迹
                        #     print("=== 准备下一次变道 ===")
                        #     new_lane_change_trajectories = scene.get_lane_change_trajectory(lateral_offset=lateral_offset)
                            
                        #     if new_lane_change_trajectories and new_lane_change_trajectories['original']:
                        #         ego_controller.set_waypoints(new_lane_change_trajectories['original'])
                        #         lane_change_stage = 0
                        #         lane_change_state = "preparing"
                        #         lane_change_trajectories = new_lane_change_trajectories
                        #         ego_controller.set_target_speed(20.0)
                        #         print(f"设置新的原始轨迹，包含 {len(new_lane_change_trajectories['original'])} 个waypoint")
                    
                    # 运行控制器
                    ego_controller.run_step()
                    
                    trajectory_check_counter += 1
                    
                except Exception as e:
                    print(f"轨迹控制运行错误: {e}")
                    print("回退到自动驾驶模式")
                    ego_controller.reset()
                    ego_controller = None
                    scene.agent.set_autopilot(True)

            if not args.no_save:  # 仅在非--no-save模式下进行记录
                if frame % step == 0:
                    print("frame:%d" % frame)
                    print("开始记录...")
                    time_start = time.time()
                    dataset = scene.record_tick()
                    dataset_save.save_datasets(dataset)
                    time_end = time.time()
                    counter += 1
                    print("记录完成！")
                    print("记录使用时间为%4fs" % (time_end - time_start))
                    print("当前记录次数：%d" % counter)
                    
                    # 显示控制器状态
                    if ego_controller and scene.agent and scene.agent.is_alive:
                        try:
                            remaining = ego_controller.get_remaining_waypoints()
                            current_speed = ego_controller.vehicle.get_velocity()
                            speed_magnitude = (current_speed.x**2 + current_speed.y**2)**0.5
                            speed_kmh = speed_magnitude * 3.6  # 转换为km/h
                            target_speed = ego_controller.target_speed
                            
                            # 变道状态信息
                            stage_names = ["原始车道", "变道中", "变道后"]
                            stage_name = stage_names[lane_change_stage] if lane_change_stage < 3 else "未知"
                            
                            print(f"变道控制状态: {lane_change_state} | 阶段: {stage_name}")
                            print(f"剩余waypoints={remaining}, 当前速度={speed_kmh:.1f}km/h, 目标速度={target_speed}km/h")
                            print(f"当前waypoint索引: {ego_controller.current_waypoint_index}")
                        except Exception as e:
                            print(f"获取控制器状态失败: {e}")
                    
                    print("*" * 60)
                    
                    if counter >= max_record:
                        print(f"达到最大记录次数{max_record}，程序即将退出...")
                        print("*" * 60)
                        print("开始合成多雷达点云...")
                        batch_merge(dataset_save.OUTPUT_FOLDER, "configs.yaml")
                        print("点云合成完成！")
                        print("*" * 60)
                        # 自动生成所有mask
                        print("开始自动生成mask...")
                        time.sleep(1)  # 等待一秒
                        # 自动生成mask并保存
                        process_all_masks(dataset_save.OUTPUT_FOLDER)
                        print("mask已生成并保存...")
                        print("*" * 60)
                        images_to_video(dataset_save.OUTPUT_FOLDER, 0, max_record, 15)
                        print("视频生成完成")
                        print("*" * 60)
                        program_end_time = time.time()
                        total_time = program_end_time - program_start_time
                        print(f"程序总运行时间：{total_time:.2f}秒")
                        break
                else:
                    # 更新场景
                    scene.update_spectator()
                    scene.world.tick()
            else:
                # --no-save模式下只更新场景
                scene.update_spectator()
                scene.world.tick()

            frame += 1
            
    finally:
        # 清理控制器
        if ego_controller:
            try:
                ego_controller.reset()
            except Exception as e:
                print(f"清理控制器时出错: {e}")
            
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()