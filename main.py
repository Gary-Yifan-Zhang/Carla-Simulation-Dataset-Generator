from utils.utils import yaml_to_config
from SimulationScene import SimulationScene
from DatasetSave import DatasetSave
import time



def main():
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

        while True:
            if frame % step == 0:
                # 记录帧
                print("frame:%d" % frame)
                print("开始记录...")
                time_start = time.time()
                dataset = scene.record_tick()
                dataset_save.save_datasets(dataset)
                time_end = time.time()
                print("记录完成！")
                print("记录使用时间为%4fs" % (time_end - time_start))
                print("********************************************************")
            else:
                # 运行帧(仅更新)
                scene.update_spectator()
                scene.world.tick()

            frame += 1
    finally:
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()
