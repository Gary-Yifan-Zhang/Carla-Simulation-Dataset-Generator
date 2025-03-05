from utils.utils import yaml_to_config
from SimulationScene import SimulationScene
from DatasetSave import DatasetSave
import time
import argparse
from utils.mask import process_all_masks



def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="仿真数据采集程序")
    parser.add_argument('--no-save', action='store_true', 
                        help="跳过数据保存步骤")
    args = parser.parse_args()

    # 加载配置文件
    config = yaml_to_config("configs.yaml")
    # 初始化仿真场景
    scene = SimulationScene(config)
    # 初始化保存设置
    dataset_save = DatasetSave(config)
    try:
        # 设置场景地图
        scene.set_map()
        # 设置场景天气
        scene.set_weather()
        # 开启同步模式
        scene.set_synchrony()
        # 在场景中生成actors(车辆与行人)
        scene.spawn_actors()
        # 设置actors自动运动
        scene.set_actors_route()
        # 生成agent（用于放置传感器的车辆与传感器）
        scene.spawn_agent()
        # 设置观察视角(与RGB相机一致)
        scene.set_spectator()
        # 监听传感器信息
        scene.listen_sensor_data()

        # 帧数
        frame = 0
        # 记录步长
        step = config["SAVE_CONFIG"]["STEP"]
        counter = 0
        # 获取最大记录次数配置（带默认值）
        max_record = config["SAVE_CONFIG"].get("MAX_RECORD_COUNT", float('inf'))

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
                        # 自动生成所有mask
                        print("开始自动生成mask...")
                        time.sleep(1)  # 等待一秒
                        # 自动生成mask并保存
                        process_all_masks(dataset_save.OUTPUT_FOLDER)
                        print("mask已生成并保存...")
                        print("*" * 60)
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
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()
