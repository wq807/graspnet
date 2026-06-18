import mujoco
import mujoco.viewer

# 纯净加载，不搞任何花里胡哨的操作
print("正在读取模型...")
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

print("✅ 模型加载成功！正在启动仿真窗口...")

# 启动视图
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
