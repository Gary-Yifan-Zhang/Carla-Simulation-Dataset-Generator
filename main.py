from utils import yaml_to_config
from SimulationScene import SimulationScene


def main():
    config = yaml_to_config("configs.yaml")
    scene = SimulationScene(config)
    try:
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
        STEP = config["SAVE_CONFIG"]["STEP"]

        while True:
            if frame % STEP == 0:
                # 记录帧
                scene.record_tick()
            else:
                # 运行帧
                scene.world.tick()
    finally:
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()
