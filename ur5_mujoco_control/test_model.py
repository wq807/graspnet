import mujoco
import numpy as np
from dm_control import mjcf

# 1. 加载机械臂和夹爪模型（路径和你本地完全匹配）
arm = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")

# 2. 正确挂载：UR5e的末端法兰站点是 tool0，不是 attachment_site
arm.find('site', 'tool0').attach(gripper)

# 3. 生成MuJoCo模型
physics = mjcf.Physics.from_mjcf_model(arm)
model = physics.model
data = physics.data

print("✅ 模型加载成功")
print(f"关节总数: {model.njnt}")
print(f"站点总数: {model.nsite}")
print("\n所有站点名称列表：")
for i in range(model.nsite):
    site_name = model.site(i).name
    print(f"  ID {i}: {site_name}")

# 4. 测试物理步进
print("\n测试物理步进...")
for _ in range(200):
    physics.step()
print("✅ 物理步进正常，无崩溃")
