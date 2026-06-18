import mujoco
import mujoco.viewer
from dm_control import mjcf

print("🚀 正在调用 DeepMind 官方工具库无损拼装模型...")

# 1. 直接读取原始的、未经任何修改的两个 XML
ur5e = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")

# 2. 核心操作：找到 UR5e 官方预留的挂载点，直接把夹爪 "咔哒" 接上去！
# (官方库会自动给所有的 material, class 加上前缀防止冲突，全自动处理)
ur5e.find('site', 'attachment_site').attach(gripper)

# 3. 添加一个地板和灯光，避免掉进黑洞
ur5e.worldbody.add('light', pos=[0, 0, 5], dir=[0, 0, -1], directional=True)
ur5e.worldbody.add('camera', name='cam_1', pos=[0.5, 0, 1.2], zaxis=[0, 0, 1])
ur5e.worldbody.add('geom', type='plane', size=[2, 2, 0.1], rgba=[0.9, 0.9, 0.9, 1])

# 4. 生成内存数据 (自动提取所有 .obj/.stl 文件的数据流，再也不怕相对路径报错)
xml_string = ur5e.to_xml_string()
assets = ur5e.get_assets()

# 5. 直接喂给 MuJoCo！
model = mujoco.MjModel.from_xml_string(xml_string, assets)
data = mujoco.MjData(model)

print("✅ 融合成功，没有任何多余的文件！正在弹出仿真器...")

# 启动窗口
import math
import time

step = 0
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        step += 1
        # 时间变量
        t = step * 0.01 
        
        # UR5e 有 6 个关节，索引 0-5
        data.ctrl[0] = math.sin(t) * 1.5      # Base (底座左右扫)
        data.ctrl[1] = -1.57 + math.sin(t)*0.5 # Shoulder (肩膀上下)
        data.ctrl[2] = 1.57                   # Elbow (手肘固定弯曲)
        # 3, 4, 5 是手腕关节，保持 0
        
        # 夹爪索引通常在最后 (索引 6)
        # 官方 2F85 夹爪的控制范围通常是 0 (全开) 到 0.8 (全闭)
        data.ctrl[6] = (math.sin(t * 5) + 1) * 0.4 
        
        # 步进物理引擎
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
