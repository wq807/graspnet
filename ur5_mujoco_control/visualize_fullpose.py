import os
import glob
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from residual_grasp_env_fullpose import ResidualGraspEnv

def get_latest_checkpoint(log_dir):
    """自动查找最新模型和对应归一化参数"""
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    model_files = glob.glob(os.path.join(checkpoints_dir, "*_steps.zip"))
    
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        step_num = latest_model.split("_")[-2]
        norm_file = os.path.join(checkpoints_dir, f"vecnormalize_{step_num}_steps.pkl")
        if os.path.exists(norm_file):
            return latest_model, norm_file
    
    best_model = os.path.join(log_dir, "best_model", "best_model.zip")
    root_norm = os.path.join(log_dir, "vec_normalize.pkl")
    return best_model, root_norm

def visualize():
    log_dir = "./logs_fullpose/"
    
    if not os.path.exists(log_dir) or not os.listdir(log_dir):
        print(f"⚠️ 未在 {log_dir} 找到数据，请先启动训练")
        return

    model_path, norm_path = get_latest_checkpoint(log_dir)
    
    print(f"📂 加载模型: {model_path}")
    print(f"📂 加载归一化参数: {norm_path}")

    env = DummyVecEnv([lambda: ResidualGraspEnv()])
    
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ 归一化参数加载成功")
    else:
        print("⚠️  未找到归一化文件，使用原始观测")

    model = PPO.load(model_path, env=env, device="cpu")
    raw_env = env.envs[0]

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        obs = env.reset()
        episode_reward = 0
        episode_step = 0
        success_count = 0
        total_episodes = 0

        print("\n🎬 开始可视化，关闭窗口即可退出")
        print("=" * 70)

        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_step += 1
            viewer.sync()

            if done[0]:
                total_episodes += 1
                is_success = info[0]["is_success"]
                if is_success:
                    success_count += 1

                # 调试信息
                tcp_pos = raw_env.data.site_xpos[raw_env.target_site_id]
                target_pos = raw_env.target_grasp_pos
                pos_err = np.linalg.norm(tcp_pos - target_pos) * 1000  # 转mm

                tcp_quat = np.empty(4)
                mujoco.mju_mat2Quat(tcp_quat, raw_env.data.site_xmat[raw_env.target_site_id].flatten())
                rot_err = 1 - np.abs(np.dot(tcp_quat, raw_env.target_grasp_quat))

                obj_qpos_idx = raw_env.model.jnt_qposadr[raw_env.target_joint_id]
                obj_pos = raw_env.data.qpos[obj_qpos_idx:obj_qpos_idx+3]
                lift = (obj_pos[2] - 0.45) * 1000  # 拔起高度mm

                gripper = raw_env.data.ctrl[6]
                has_contact = raw_env._is_gripper_contact_object()

                print(f"回合 {total_episodes:2d} | 步数: {episode_step:3d} | 奖励: {episode_reward:6.1f}")
                print(f"  位置误差: {pos_err:.1f}mm | 姿态误差: {rot_err:.3f} | 拔起高度: {lift:.1f}mm")
                print(f"  夹爪开度: {gripper:.2f} | 接触状态: {'✅ 有' if has_contact else '❌ 无'}")
                print(f"  是否成功: {'✅ 是' if is_success else '❌ 否'} | 累计成功率: {success_count/total_episodes:.2%}")
                print("-" * 70)

                obs = env.reset()
                episode_reward = 0
                episode_step = 0

if __name__ == "__main__":
    visualize()
