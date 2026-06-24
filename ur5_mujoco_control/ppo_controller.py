import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import mujoco

# 借用环境结构加载归一化参数
from residual_grasp_env import ResidualGraspEnv 

class PPOController:
    def __init__(self, log_dir="./logs_residual/"):
        model_path = os.path.join(log_dir, "best_model", "best_model_v1.zip")
        norm_path = os.path.join(log_dir, "vec_normalize_v1.pkl")
        
        print("🔧 [PPO] 正在装载柔顺微操小脑...")
        self.dummy_env = DummyVecEnv([lambda: ResidualGraspEnv()])
        if os.path.exists(norm_path):
            self.dummy_env = VecNormalize.load(norm_path, self.dummy_env)
            self.dummy_env.training = False      
            self.dummy_env.norm_reward = False   
        else:
            raise FileNotFoundError("找不到 vec_normalize.pkl，PPO 会瞎掉！")

        self.model = PPO.load(model_path, env=self.dummy_env, device="cpu")
        print("✅ [PPO] 小脑装载完毕，随时准备接管！")

    def get_action(self, obs_array):
        # deterministic=True 保证动作稳定不乱抖
        action, _states = self.model.predict(obs_array, deterministic=True)
        return action
