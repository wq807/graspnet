import mujoco
import mujoco.viewer
from dm_control import mjcf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import zmq
import cv2

print("🚀 [1/5] 正在构建物理世界：正面斜上方第一视角...")

# ==========================================
# 1. 拼装机器人与环境 (完全保留你的环境)
# ==========================================
world = mjcf.RootElement()
world.worldbody.add('light', pos=[0, 0, 5], dir=[0, 0, -1], directional=True)
world.worldbody.add('geom', name='floor', type='plane', size=[2, 2, 0.1], rgba=[0.9, 0.9, 0.9, 1])

table = world.worldbody.add('body', name='workbench', pos=[0, 0, 0.2])
table.add('geom', type='box', size=[0.6, 0.35, 0.2], rgba=[0.65, 0.55, 0.45, 1])

ur5e = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")
ur5e.find('site', 'attachment_site').attach(gripper)
arm_mount = world.worldbody.add('site', name='arm_mount', pos=[-0.25, 0, 0.4])
arm_mount.attach(ur5e)
# --- 手眼相机 (Eye-in-Hand) 挂载配置 ---
# 1. 查找机械臂末端手腕刚体
wrist = ur5e.find('body', 'wrist_3_link') 
if wrist is None:
    wrist = ur5e.find('body', 'ur5e_wrist_3_link')

# 2. 将手眼相机物理拼装到手腕上 (移除了所有可能引发编译错误的非法字符)
wrist.add('camera', name='eye_in_hand_cam', pos=[0, 0.08, 0.05], fovy=60, xyaxes=[1, 0, 0, 0, 0, 1])
print(">>> [OK] Eye-in-Hand camera successfully mounted to wrist_3_link.")

# 替换原来的 target_block 部分
target_x = random.uniform(0.15, 0.35)
target_y = random.uniform(-0.15, 0.15)
target_body = world.worldbody.add('body', name='target_cylinder', pos=[target_x, target_y, 0.45])
target_body.add('joint', type='free')
# 半径3cm，高度5cm，和易拉罐尺寸接近
target_body.add('geom', type='cylinder', size=[0.03, 0.05], rgba=[0.2, 0.5, 0.8, 1], mass=0.1)


world.worldbody.add('camera', name='workspace_cam', pos=[0.8, 0, 0.95], fovy=45, xyaxes=[0, 1, 0, -0.707, 0, 0.707])

# ==========================================
# 2. 编译物理引擎并执行预备动作
# ==========================================
print("⏳ [2/5] 物理引擎预热...")
model = mujoco.MjModel.from_xml_string(world.to_xml_string(), world.get_assets())
data = mujoco.MjData(model)

home_pose = [3.14, -1.57, 1.57, -1.57, -1.57, 0]
data.ctrl[:6] = home_pose

for _ in range(500):
    mujoco.mj_step(model, data)

# ==========================================
# 3. 提取视觉数据并投影 3D 点云
# ==========================================
print("📸 [3/5] 正在拍照...")
width, height = 640, 480
renderer = mujoco.Renderer(model, height=height, width=width)

renderer.update_scene(data, camera='workspace_cam')
rgb_image = renderer.render()

renderer.enable_depth_rendering()
renderer.update_scene(data, camera='workspace_cam')
depth_image = renderer.render()
renderer.disable_depth_rendering()

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'workspace_cam')
fovy = model.cam_fovy[cam_id]
f = 0.5 * height / np.tan(fovy * np.pi / 360)
cx, cy = width / 2, height / 2

u, v = np.meshgrid(np.arange(width), np.arange(height))
Z = depth_image
# 生成标准的 OpenCV 相机系点云 (+Z 朝前，+X 朝右，+Y 朝下)
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
point_cloud_cv = np.dstack((X, Y, Z)).reshape(-1, 3)

# ==========================================
# 4. 终极精准点云过滤 (自动适配世界坐标)
# ==========================================
# 直接从底层读取相机位姿，彻底避免手动算错矩阵！
cam_pos = data.cam_xpos[cam_id]
R_mj2world = data.cam_xmat[cam_id].reshape(3, 3)

# 核心修正：MuJoCo相机到OpenCV相机的标准变换矩阵
R_cv2mj = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
# 最终的 OpenCV相机 -> 真实世界 矩阵
R_cv2world = R_mj2world @ R_cv2mj

# 步骤1：基础滤除无深度点
valid_depth = (point_cloud_cv[:, 2] > 0.1) & (point_cloud_cv[:, 2] < 2.0)
pc_cv_valid = point_cloud_cv[valid_depth]
print(f"   基础过滤后点数: {len(pc_cv_valid)}")

# 步骤2：转换到真实世界坐标系进行空间切割
pc_world = (R_cv2world @ pc_cv_valid.T).T + cam_pos

# 步骤3：极其严格的“3D抓取工作区”裁剪 (Workspace Crop)
# 方块的活动范围大概是 X:[0.15, 0.35], Y:[-0.15, 0.15], 落在 Z=0.4 的桌面上
obj_mask = (
    (pc_world[:, 0] > 0.05) & (pc_world[:, 0] < 0.45) &   # X 轴限制：彻底把左侧的机械臂一刀切掉！
    (pc_world[:, 1] > -0.25) & (pc_world[:, 1] < 0.25) &  # Y 轴限制：只留桌子正中区域
    (pc_world[:, 2] > 0.402) & (pc_world[:, 2] < 0.6)     # Z 轴限制：贴着桌面切，不要桌子，只要悬空物体
)
pc_cv_final = pc_cv_valid[obj_mask]
print(f"   严格 3D 工作区裁剪后点数: {len(pc_cv_final)} (现在绝对只剩方块了！)")

# 步骤4：标准化采样
num_points = 20000
if len(pc_cv_final) == 0:
    print("⚠️ 警告：有效点为空！用原始点云兜底。")
    pc_cv_final = pc_cv_valid

if len(pc_cv_final) >= num_points:
    idx = np.random.choice(len(pc_cv_final), num_points, replace=False)
    pc_sample = pc_cv_final[idx]
else:
    idx = np.random.choice(len(pc_cv_final), num_points, replace=True)
    pc_sample = pc_cv_final[idx]

# ==========================================
# 5. 发送至服务器 (携带世界矩阵，让服务器拥有“上帝视角”)
# ==========================================
print("📡 [4/5] 发送点云至 GraspNet...")
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# 将点云和转换矩阵一起打包发送
payload = {
    "point_cloud": pc_sample.astype(np.float32),
    "R_cv2world": R_cv2world.astype(np.float32) 
}
socket.send_pyobj(payload)

best_grasp = socket.recv_pyobj()
socket.close()
context.term()

if best_grasp["success"]:
    print(f"🎉 [5/5] 收到最优抓取位姿！分数: {best_grasp['score']:.4f}")
    
    grasp_center = np.array(best_grasp["translation"])
    rot_mat = np.array(best_grasp["rotation_matrix"])
    grasp_width = best_grasp["width"]
    grasp_depth = best_grasp["depth"]
    
# ==========================================
# ==========================================
    # 🧠 [里程臂 2] 核心：坐标系转换 (Camera -> World -> Robot Base)
    # ==========================================
    # 1. 将 GraspNet 在相机坐标系下的抓取中心，转换到世界绝对坐标系 (World Frame)
    # 公式: P_world = R_cv2world * P_cam + T_cam
    grasp_center_world = R_cv2world @ grasp_center + cam_pos
    R_grasp_world = R_cv2world @ rot_mat

    # 2. 将世界绝对坐标，转换到机械臂底座坐标系 (Robot Base Frame)
    # 依据你的代码: arm_mount = world.worldbody.add('site', name='arm_mount', pos=[-0.25, 0, 0.4])
    # 机械臂基座中心在世界坐标中绝对位置是 [-0.25, 0.0, 0.4]，且方向与世界坐标轴完全平行
    base_pos = np.array([-0.25, 0.0, 0.4]) 
    
    # 减去底座偏移量，即完成从世界系到机械臂底座系的平移
    grasp_center_base = grasp_center_world - base_pos
    R_grasp_base = R_grasp_world  # 旋转矩阵保持相同，因为基座和世界坐标轴对齐

    # 3. 计算预抓取悬停点 (Pre-grasp Pose)
    # 在 GraspNet 夹爪定义中，+Z 轴的相反方向（-Z）是夹爪向前伸出的方向
    # 因此，向后退 10 厘米（0.10米）作为预备悬停点，就是沿着本地的 +Z 轴平移
    approach_vector = R_grasp_base[:, 2] # 提取 Z 轴方向向量
    pre_grasp_pos = grasp_center_base + approach_vector * 0.10

    print("\n=========================================")
    print("📍 [里程碑 2 通关] 机械臂目标位姿 (Base Frame):")
    print(f"   🤖 最终抓取点 XYZ: [{grasp_center_base[0]:.4f}, {grasp_center_base[1]:.4f}, {grasp_center_base[2]:.4f}]")
    print(f"   🚁 预备悬停点 XYZ: [{pre_grasp_pos[0]:.4f}, {pre_grasp_pos[1]:.4f}, {pre_grasp_pos[2]:.4f}]")
    print("=========================================\n")
    # ==========================================
    # 🌟 1. 生成 2D 升级版立体线框图并保存
    # ==========================================
    def project(pt):
        if pt[2] <= 1e-6: return None
        return (int(f * pt[0] / pt[2] + cx), int(f * pt[1] / pt[2] + cy))

    # 构造带有“厚度”的 8 顶点 3D 包围盒
    hw = grasp_width / 2.0
    hd = grasp_depth
    hh = 0.02  # 虚拟手指厚度
    vertices_local = np.array([
        [hw, hh, 0], [hw, -hh, 0], [-hw, -hh, 0], [-hw, hh, 0],
        [hw, hh, -hd], [hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd]
    ])
    vertices_cam = (rot_mat @ vertices_local.T).T + grasp_center
    pts_2d = [project(p) for p in vertices_cam]

    rgb_draw = rgb_image.copy()
    if None not in pts_2d:
        color = (0, 0, 255) # BGR 红色
        thick = 3
        # 画上表面和下表面
        for i in range(4):
            cv2.line(rgb_draw, pts_2d[i], pts_2d[(i+1)%4], color, thick)
            cv2.line(rgb_draw, pts_2d[i+4], pts_2d[(i+1)%4+4], color, thick)
            cv2.line(rgb_draw, pts_2d[i], pts_2d[i+4], color, thick)
        
        # 画蓝色中心引导线
        center_2d = project(grasp_center)
        base_cam = rot_mat @ np.array([0, 0, -hd*1.5]) + grasp_center
        base_2d = project(base_cam)
        if center_2d and base_2d:
            cv2.line(rgb_draw, center_2d, base_2d, (255, 0, 0), thick + 1)

    # 保存 2D 图片到本地
    cv2.imwrite("grasp_result_2d.jpg", cv2.cvtColor(rgb_draw, cv2.COLOR_RGB2BGR))
    print("🖼️ 2D抓取结果图已保存：./grasp_result_2d.jpg")

    # ==========================================
    # 🌟 2. 生成 3D 点云空间可视化图并保存
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Best Grasp Pose on 3D Point Cloud")

    # 降采样点云以加快绘图速度
    viz_pc = pc_sample[::5] 
    ax.scatter(viz_pc[:, 0], viz_pc[:, 1], viz_pc[:, 2], c='gray', s=1, alpha=0.5, label='Scene Point Cloud')

    # 计算 3D 抓取线条
    p1 = grasp_center + rot_mat @ np.array([hw, 0, 0])
    p2 = grasp_center + rot_mat @ np.array([-hw, 0, 0])
    p3 = p1 + rot_mat @ np.array([0, 0, -hd])
    p4 = p2 + rot_mat @ np.array([0, 0, -hd])
    p_base_center = grasp_center + rot_mat @ np.array([0, 0, -hd])
    p_approach = p_base_center + rot_mat @ np.array([0, 0, -hd*0.5])

    # 绘制 3D 抓取部件
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='r', linewidth=3, label='Grasp Opening')
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]], c='g', linewidth=3, label='Grasp Base')
    ax.plot([p2[0], p4[0]], [p2[1], p4[1]], [p2[2], p4[2]], c='g', linewidth=3)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], c='g', linewidth=3)
    ax.plot([p_base_center[0], p_approach[0]], [p_base_center[1], p_approach[1]], [p_base_center[2], p_approach[2]], c='b', linewidth=3, label='Approach Direction')
    ax.scatter(grasp_center[0], grasp_center[1], grasp_center[2], c='r', marker='*', s=100, label='Grasp Center')

    ax.set_xlabel('X (Camera Frame, m)')
    ax.set_ylabel('Y (Camera Frame, m)')
    ax.set_zlabel('Z (Camera Frame, m)')
    ax.legend()

    # 保存 3D 图片到本地
    plt.savefig("grasp_result_3d.jpg", dpi=300, bbox_inches='tight')
    print("🖼️ 3D抓取结果图已保存：./grasp_result_3d.jpg")

    # 同时在屏幕上弹出这两张图
    plt.show()
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:6] = home_pose
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
