import os
import glob
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from residual_grasp_env import ResidualGraspEnv

def make_env(rank, seed=0):
    def _init():
        env = ResidualGraspEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    num_cpu = 6
    log_dir = "./logs_residual/"
    os.makedirs(log_dir, exist_ok=True)
    norm_path = os.path.join(log_dir, "vec_normalize.pkl")

    print(f"🚀 正在搭建并行训练架构，开启 {num_cpu} 个环境...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)

    # 断点续训配置
    best_path = os.path.join(log_dir, "best_model", "best_model.zip")
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    model = None

    # 微调参数与新训练参数保持一致
    fine_tune_args = {
        "learning_rate": 2e-5,
        "ent_coef": 0.03,
        "n_epochs": 5,
        "clip_range": 0.12
    }

    # 优先加载最新检查点
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        try:
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                eval_env = VecNormalize.load(norm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            model = PPO.load(latest_checkpoint, env=vec_env, custom_objects=fine_tune_args)
            print("✅ 断点续训准备就绪")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}")
            model = None

    # 次优加载最优模型
    if model is None and os.path.exists(best_path):
        try:
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                eval_env = VecNormalize.load(norm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            model = PPO.load(best_path, env=vec_env, custom_objects=fine_tune_args)
            print("✅ 从 Best Model 开始续训")
        except Exception as e:
            print(f"⚠️ Best Model 加载失败: {e}")
            model = None

    # 全新初始化模型
    if model is None:
        print("🌱 初始化全新柔顺抓取模型...")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=5.0,
            gamma=0.95
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False
        )

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=2e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.92,
            clip_range=0.12,
            ent_coef=0.03,
            max_grad_norm=0.5,
            device="cpu",
            tensorboard_log=log_dir,
            verbose=1
        )

    # 回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=3000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_grasp",
        save_vecnormalize=True
    )

    all_callbacks = CallbackList([eval_callback, checkpoint_callback])

    print(f"✅ 模型运行设备: {model.device}")
    print("🔥 强化学习训练正式启动...")
    model.learn(total_timesteps=1000000, callback=all_callbacks, reset_num_timesteps=False)

    latest_path = os.path.join(log_dir, "latest_model.zip")
    model.save(latest_path)
    vec_env.save(norm_path)
    print(f"🏁 训练完成！模型已保存至 {latest_path}")
