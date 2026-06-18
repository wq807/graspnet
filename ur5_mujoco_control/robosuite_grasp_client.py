import numpy as np
import zmq
import robosuite as suite
import robosuite.controllers

def main():
    # 1. 通信初始化
    context = zmq.Context()
    client_socket = context.socket(zmq.REQ)
    client_socket.connect("tcp://localhost:5555")
    print("📡 [通信已就绪]")

    # 2. 【硬编码配置】完全避开任何控制器的注册表查找
    # JOINT_POSITION 是 robosuite 的基础，无需额外加载项，绝对不会报 AssertionError
    controller_config = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "control_delta": True,
        "interpolation": None,
    }

    # 3. 初始化环境
    try:
        print("🌍 [正在启动环境...]")
        env = suite.make(
            env_name="PickPlace", 
            robots="UR5e",
            controller_configs=controller_config,
            has_renderer=True,
            camera_names="frontview",
        )
    except Exception as e:
        print(f"❌ 环境启动失败: {e}")
        return

    obs = env.reset()
    print("🌍 [环境就绪]")

    # 使用 JOINT_POSITION 控制时，动作空间通常是 7 维 (6个关节 + 1个夹爪)
    gripper_state = -1 
    
    try:
        for i in range(1000):
            # A. 感知数据获取
            depth = obs["frontview_depth"]
            height, width = depth.shape
            cam_id = env.sim.model.camera_name2id("frontview")
            fovy = env.sim.model.cam_fovy[cam_id]
            f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
            
            xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
            z = depth * 2.0 
            mask = (z < 1.5) & (z > 0.05)
            x = (xmap[mask] - width/2) * z[mask] / f
            y = (ymap[mask] - height/2) * z[mask] / f
            cam_pos = env.sim.data.cam_xpos[cam_id]
            cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3)
            points = np.stack([x, y, z[mask]], axis=-1) @ cam_mat.T + cam_pos
            
            # B. 通信与动作决策
            client_socket.send_pyobj(points)
            response = client_socket.recv_pyobj()
            
            # 初始化动作数组 (7维: 6关节位置 + 1夹爪)
            # JOINT_POSITION 控制器接受的是目标关节角度偏移量
            action = np.zeros(7)
            action[6] = gripper_state 
            
            if response.get("success"):
                print("🎯 [感知到目标] 正在尝试调整...")
                # 这里我们简单做：当感知到目标时，稍微向目标方向微调一下关节角度
                # (真正的 IK 逻辑需要以后在服务器端计算，现在先保证能跑通环境)
                action[:6] = 0.01 
            
            obs, reward, done, info = env.step(action)
            env.render()
            
    except Exception as e:
        print(f"❌ 运行报错: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
