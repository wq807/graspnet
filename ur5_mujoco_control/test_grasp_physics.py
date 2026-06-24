import numpy as np
import mujoco
import mujoco.viewer
from residual_grasp_env import ResidualGraspEnv, solve_ik_scipy

def test_grasp_physics():
    # 1. 初始化环境（和训练完全一致的模型与物理参数）
    env = ResidualGraspEnv()
    model = env.model
    data = env.data

    # 2. 复位环境，获取真实物体位置
    obs, _ = env.reset()
    obj_qpos_idx = model.jnt_qposadr[env.target_joint_id]
    real_obj_pos = data.qpos[obj_qpos_idx:obj_qpos_idx+3].copy()
    
    print("✅ 环境复位完成，机械臂已到达预抓取位")
    print(f"初始TCP位置: {data.site_xpos[env.target_site_id]}")
    print(f"初始物体位置: {real_obj_pos}")

    # 3. 定义抓取关键位姿（基于真实物体位置计算，保证对齐）
    R_down = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 末端朝下的旋转矩阵

    # 阶段1目标：物体正上方，抓取高度（物体质心中部）
    grasp_pos = real_obj_pos.copy()
    grasp_pos[2] = real_obj_pos[2]  # 和物体质心同高，夹爪夹持物体中部

    # 阶段2目标：抬起后的位置（向上抬10cm）
    lift_pos = real_obj_pos.copy()
    lift_pos[2] = real_obj_pos[2] + 0.1

    # 4. 求解各阶段的逆运动学
    q_grasp = solve_ik_scipy(model, data, 'tcp_site', grasp_pos, R_down)
    q_lift = solve_ik_scipy(model, data, 'tcp_site', lift_pos, R_down)

    # 5. 启动可视化窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 调整相机视角，方便观察
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45

        # ====================== 阶段0：保持预抓取位，等待稳定 ======================
        print("\n⏳ 阶段0：保持预抓取位，等待物理稳定...")
        for _ in range(200):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_state(model, data, env)

        # ====================== 阶段1：缓慢下探到抓取高度 ======================
        print("\n⬇️  阶段1：下探到抓取高度")
        # 平滑插值移动，避免冲击碰倒物体
        q_start = data.ctrl[:6].copy()
        steps = 300
        for i in range(steps):
            alpha = i / steps
            data.ctrl[:6] = q_start * (1 - alpha) + q_grasp * alpha
            mujoco.mj_step(model, data)
            viewer.sync()
        print_state(model, data, env)

        # ====================== 阶段2：闭合夹爪 ======================
        print("\n🤏 阶段2：闭合夹爪")
        gripper_close = 1.0  # 完全闭合，可根据模型实际范围调整0~1
        steps = 200
        start_gripper = data.ctrl[6]
        for i in range(steps):
            alpha = i / steps
            data.ctrl[6] = start_gripper * (1 - alpha) + gripper_close * alpha
            mujoco.mj_step(model, data)
            viewer.sync()
        # 保持夹紧100步，观察夹持状态
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_state(model, data, env)

        # ====================== 阶段3：缓慢抬起 ======================
        print("\n⬆️  阶段3：向上抬起物体")
        q_start_lift = data.ctrl[:6].copy()
        steps = 400
        for i in range(steps):
            alpha = i / steps
            data.ctrl[:6] = q_start_lift * (1 - alpha) + q_lift * alpha
            mujoco.mj_step(model, data)
            viewer.sync()
        print_state(model, data, env)

        # ====================== 阶段4：保持抬起，观察稳定性 ======================
        print("\n⏸️  阶段4：保持抬起状态，观察是否滑落...")
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_state(model, data, env)

        print("\n🏁 测试完成！关闭可视化窗口即可退出。")
        while viewer.is_running():
            viewer.sync()

def print_state(model, data, env):
    """打印当前关键状态，辅助排查问题"""
    tcp_pos = data.site_xpos[env.target_site_id]
    obj_qpos_idx = model.jnt_qposadr[env.target_joint_id]
    obj_pos = data.qpos[obj_qpos_idx:obj_qpos_idx+3]
    obj_quat = data.qpos[obj_qpos_idx+3:obj_qpos_idx+7]
    
    # 计算物体倾斜角度
    rot_mat = np.empty(9)
    mujoco.mju_quat2Mat(rot_mat, obj_quat)
    body_z = rot_mat[2::3]
    tilt_angle = np.arccos(np.clip(np.dot(body_z, [0,0,1]), -1, 1)) * 180 / np.pi
    
    gripper = data.ctrl[6]
    
    print(f"  TCP位置: ({tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f})")
    print(f"  物体位置: ({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
    print(f"  物体倾斜: {tilt_angle:.2f}°")
    print(f"  夹爪开度指令: {gripper:.3f}")

if __name__ == "__main__":
    test_grasp_physics()
