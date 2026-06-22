import sys
import os

# 1. 强制清理环境变量，防止 PYTHONPATH 干扰
if "PYTHONPATH" in os.environ:
    del os.environ["PYTHONPATH"]

# 2. 指定你刚刚找到的正确路径
correct_path = '/home/wq/ur_graspnet/ur5_mujoco_control/ompl/build/py-bindings'
sys.path.insert(0, correct_path)

# 3. 强制导入并打印调试信息
try:
    import ompl
    print("【成功】OMPL 导入路径:", ompl.__file__)
    from ompl import base as ob
    print("【成功】Base 模块加载成功！")
except Exception as e:
    print("【失败】具体错误:")
    print(e)
    # 打印 sys.path 看看 Python 到底在搜哪里
    print("\n当前搜索路径:", sys.path)
