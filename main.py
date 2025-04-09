from utils.utils import yaml_to_config
from SimulationScene import SimulationScene
from DatasetSave import DatasetSave
import time
import argparse
from utils.mask import process_all_masks
from utils.visual import images_to_video
from utils.pointcloud import batch_merge


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="仿真数据采集程序")
    parser.add_argument('--no-save', action='store_true', 
                        help="跳过数据保存步骤")
    parser.add_argument('--scenario-name', type=str, default="UnknownScenario",
                        help="当前运行的场景名称")
    args = parser.parse_args()
    program_start_time = time.time()

    # 加载配置文件
    config = yaml_to_config("configs.yaml")
    # 初始化仿真场景
    scene = SimulationScene(config)
    if not args.no_save:
        # 初始化保存设置
        dataset_save = DatasetSave(config, args.scenario_name)
    try:
        # 设置场景地图
        scene.set_map()
        # # 执行场景
        # scene.run_scenario()   
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
        INIT_WAIT_FRAMES = 10  # 等待100帧（约5秒，假设20FPS）
        for _ in range(INIT_WAIT_FRAMES):
            scene.update_spectator()  # 保持视角更新
            scene.world.tick()
            print(f"预运行进度: {_+1}/{INIT_WAIT_FRAMES}", end='\r')
        print("\n预运行完成，开始主循环")

        while True:
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
                # scene.update_spectator()
                scene.world.tick()

            frame += 1
    finally:
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()
