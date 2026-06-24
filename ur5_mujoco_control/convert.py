import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from residual_grasp_env import ResidualGraspEnv

print("🔄 开始加载原版 Numpy 2.x 模型...")
env = DummyVecEnv([lambda: ResidualGraspEnv()])
# 强行加载旧的归一化文件
env = VecNormalize.load("./logs_residual/vec_normalize.pkl", env)
# 强行加载旧的模型
model = PPO.load("./logs_residual/best_model/best_model.zip", env=env)

print("💾 正在使用当前环境重新打包导出...")
# 另存为 _v1 格式
model.save("./logs_residual/best_model/best_model_v1.zip")
env.save("./logs_residual/vec_normalize_v1.pkl")

print("✅ 模型转译大功告成！快去看看文件夹里有没有 _v1 的文件！")
