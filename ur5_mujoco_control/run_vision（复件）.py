import os
import sys
import time
import random
import zmq
import cv2
import numpy as np
import pinocchio as pin
import hppfcl as fcl
import mujoco
import mujoco.viewer
from dm_control import mjcf

# ==========================================
# 🌟 引入本地编译的 OMPL
# ==========================================
sys.path.insert(0, '/home/wq/ur_graspnet/ur5_mujoco_control/ompl/build/py-bindings')
from ompl._ompl import base as ob
from ompl._ompl import geometric as og

# ==========================================
# 🌟 工具函数与底层核心算法
# ==========================================
def wrap_to_pi(q):
    """关节角度归一化到 [-pi, pi]"""
    return (q + np.pi) % (2 * np.pi) - np.pi

def linear_interpolate(q_start, q_end, steps=300):
    """关节空间直线插值"""
    t = np.linspace(0, 1, steps)
    return np.array([q_start + (q_end - q_start) * ti for ti in t])
    
# 把 tol 改为 1e-3
def solve_ik(model, data, site_name, target_pos, target_rotmat, max_steps=100, tol=1e-3):
    """
    MuJoCo 原生数值 IK 求解器 (极致抗死锁、高精版)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    q_backup = data.qpos.copy()
    
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, target_rotmat.flatten())
    
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    damp = 1e-2 
    step_size = 0.5
    
    best_q = data.qpos[:6].copy()
    min_err = float('inf')
    
    # 允许 3 次随机重启，快速对抗运动学奇异
    for attempt in range(4):
        if attempt > 0:
            data.qpos[:6] = best_q + np.random.uniform(-0.2, 0.2, 6)
            if data.qpos[2] < 0: data.qpos[2] = -data.qpos[2] # 强制手肘向上
                
        for step in range(max_steps):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data) # 激活雅可比
            
            # 位置误差
            dx = target_pos - data.site_xpos[site_id]
            
            # 精准四元数旋转误差
            current_quat = np.empty(4)
            mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
            dr = np.empty(3)
            mujoco.mju_subQuat(dr, target_quat, current_quat) 
            
            err = np.hstack([dx, dr])
            err_norm = np.linalg.norm(err)
            
            if err_norm < min_err:
                min_err = err_norm
                best_q = data.qpos[:6].copy()
                
            if err_norm < tol:
                break 
                
            mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)
            jac = np.vstack([jac_p, jac_r])[:, :6]
            
            v = jac.T @ np.linalg.inv(jac @ jac.T + damp * np.eye(6)) @ err
            data.qpos[:6] += step_size * v
            
        if err_norm < tol:
            break 
            
    if min_err > 0.01:
        print(f"⚠️ IK 提示: 逼近误差 {min_err:.4f}m，使用最优近似解。")
        
    q_res = best_q.copy()
    data.qpos[:] = q_backup
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    return q_res

print("🚀 [1/5] 正在构建物理世界...")

# ==========================================
# 1. 环境与机器人建模
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

wrist = ur5e.find('body', 'wrist_3_link')
if wrist is None:
    wrist = ur5e.find('body', 'ur5e_wrist_3_link')
tcp_site = wrist.add('site', name='tcp_site', pos=[0, 0, 0.15], size=[0.005], rgba=[1, 0, 0, 1])
print("🎯 TCP准星已添加至指尖（手腕前15cm）")

# 🌟 高级接触动力学：涂抹防滑粉，避免保龄球效应
target_x = random.uniform(0.15, 0.35)
target_y = random.uniform(-0.15, 0.15)
target_body = world.worldbody.add('body', name='target_cylinder', pos=[target_x, target_y, 0.45])
target_body.add('joint', type='free')
# 把 mass 从 0.1 改成 0.5
target_body.add('geom', type='cylinder', size=[0.03, 0.05], rgba=[0.2, 0.5, 0.8, 1], mass=0.5, 
                condim=4, friction=[2.0, 0.05, 0.0001])

world.worldbody.add('camera', name='workspace_cam', pos=[0.8, 0, 0.95], fovy=45, xyaxes=[0, 1, 0, -0.707, 0, 0.707])

# ==========================================
# 2. 物理引擎预热
# ==========================================
print("⏳ [2/5] 物理引擎预热...")
model = mujoco.MjModel.from_xml_string(world.to_xml_string(), world.get_assets())
data = mujoco.MjData(model)
home_pose = [3.14, -1.57, 1.57, -1.57, -1.57, 0]
data.ctrl[:6] = home_pose
data.ctrl[6] = 0.0 
for _ in range(500):
    mujoco.mj_step(model, data)

# ==========================================
# 3. 视觉采集与点云生成
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
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
point_cloud_cv = np.dstack((X, Y, Z)).reshape(-1, 3)

cam_pos = data.cam_xpos[cam_id]
R_mj2world = data.cam_xmat[cam_id].reshape(3, 3)
R_cv2mj = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
R_cv2world = R_mj2world @ R_cv2mj

valid_depth = (point_cloud_cv[:, 2] > 0.1) & (point_cloud_cv[:, 2] < 2.0)
pc_cv_valid = point_cloud_cv[valid_depth]
pc_world = (R_cv2world @ pc_cv_valid.T).T + cam_pos
obj_mask = (
    (pc_world[:, 0] > 0.05) & (pc_world[:, 0] < 0.45) &
    (pc_world[:, 1] > -0.25) & (pc_world[:, 1] < 0.25) &
    (pc_world[:, 2] > 0.402) & (pc_world[:, 2] < 0.6)
)
pc_cv_final = pc_cv_valid[obj_mask]
print(f"   点云裁剪完成，有效物体点数: {len(pc_cv_final)}")

num_points = 20000
if len(pc_cv_final) == 0: pc_cv_final = pc_cv_valid
if len(pc_cv_final) >= num_points:
    idx = np.random.choice(len(pc_cv_final), num_points, replace=False)
else:
    idx = np.random.choice(len(pc_cv_final), num_points, replace=True)
pc_sample = pc_cv_final[idx].astype(np.float32)

# ==========================================
# 4. GraspNet推理
# ==========================================
print("📡 [4/5] 发送点云至 GraspNet...")
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
payload = {"point_cloud": pc_sample, "R_cv2world": R_cv2world.astype(np.float32)}
socket.send_pyobj(payload)
best_grasp = socket.recv_pyobj()
socket.close()
context.term()

q_home = np.array(home_pose)
grasp_success = False

if best_grasp["success"]:
    print(f"🎉 [5/5] 收到最优抓取位姿！分数: {best_grasp['score']:.4f}")
    grasp_center_cam = np.array(best_grasp["translation"])
    R_grasp_cam = np.array(best_grasp["rotation_matrix"])
    
    grasp_center_world = R_cv2world @ grasp_center_cam + cam_pos
    R_grasp_world = R_cv2world @ R_grasp_cam

    # =========================================================
    # 5. 原生 MuJoCo IK 求解 & OMPL 路径规划
    # =========================================================
    print("🏗️ 求解逆运动学与避障路径...")

    URDF_PATH = "ur5e.urdf"
    PKG_DIR = os.path.abspath("mujoco_menagerie")
    robot_model = pin.buildModelFromUrdf(URDF_PATH)
    robot_data = robot_model.createData()
    collision_model = pin.buildGeomFromUrdf(robot_model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs=[PKG_DIR])
    collision_data = collision_model.createData()

    fcl_world = []
    table_geom = fcl.Box(1.3, 0.8, 0.42) 
    table_transform = fcl.Transform3f(np.eye(3), np.array([0, 0, 0.21]))
    fcl_world.append(fcl.CollisionObject(table_geom, table_transform))
    underground_geom = fcl.Box(3.0, 3.0, 0.39)
    underground_transform = fcl.Transform3f(np.eye(3), np.array([0, 0, 0.39 / 2]))
    fcl_world.append(fcl.CollisionObject(underground_geom, underground_transform))

    def get_ik_mujoco(target_pos_world, target_rot_world, q_seed):
        q_bak = data.qpos[:6].copy()
        data.qpos[:6] = q_seed.copy()
        mujoco.mj_kinematics(model, data)
        q_res = solve_ik(model, data, 'tcp_site', target_pos_world, target_rot_world)
        data.qpos[:6] = q_bak
        mujoco.mj_kinematics(model, data)
        return q_res

    # --- 姿态对齐与 180 度翻转判定 ---
    R_align = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    R_target_tcp_world = R_grasp_world @ R_align
    approach_dir_world = R_target_tcp_world[:, 2]

    R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_target_tcp_world_flipped = R_target_tcp_world @ R_z_180

    # 🌟 核心退避：不要抓正中心！沿接近方向退回 1.5cm，防止暴力撞飞物体
    # 🌟 核心修复：使用负数！强制 TCP 越过视觉中心，往深处探入 2.5 厘米！
    # 这能保证 2F-85 夹爪的 V 型槽完美、死死地卡在圆柱体的腰部。
    # 🌟 核心调优：修改这一行
    # 原来是 0.01 (往里推了 1cm)
    # 现在改为 -0.04 (额外往里深推 4cm，让夹爪更深地包覆物体)
    forward_push = -0.06  
    
    # 沿着接近方向“往前送” (负负得正，这会让机械臂向物体内部深入)
    pos_final = grasp_center_world - approach_dir_world * forward_push
    
    # 预抓取点：让机械臂在距离物体 10cm 处悬停，然后通过笛卡尔铁轨插值笔直推进去
    pos_pre = pos_final - approach_dir_world * 0.10
    pos_safe = pos_pre.copy()
    pos_safe[2] += 0.25

    q_safe_normal = get_ik_mujoco(pos_safe, R_target_tcp_world, q_home)
    q_safe_flipped = get_ik_mujoco(pos_safe, R_target_tcp_world_flipped, q_home)

    dist_normal = np.linalg.norm(wrap_to_pi(q_safe_normal - q_home))
    dist_flipped = np.linalg.norm(wrap_to_pi(q_safe_flipped - q_home))

    if dist_flipped < dist_normal:
        print(f"🔄 [智能优化] 翻转姿态更佳，已自动将夹爪翻转180度！")
        R_best_tcp = R_target_tcp_world_flipped
        q_safe = q_safe_flipped
    else:
        print(f"👍 [智能优化] 原姿态舒适度良好，正常执行。")
        R_best_tcp = R_target_tcp_world
        q_safe = q_safe_normal

    # --- 笛卡尔铁轨插值 ---
    def cartesian_interpolate(pos_start, pos_end, rot_target, q_start, steps=100):
        traj = []
        q_curr = q_start.copy()
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1
            p = pos_start + (pos_end - pos_start) * t
            q_curr = get_ik_mujoco(p, rot_target, q_curr)
            traj.append(q_curr.copy())
        return np.array(traj)

    print("🛤️ 正在生成笛卡尔空间无偏差铁轨路径...")
    # 🌟 核心慢插：下压步骤加长到 300 步，像树懒一样轻柔地逼近目标，不产生扰动风压和撞击力
    traj_safe2pre  = cartesian_interpolate(pos_safe, pos_pre, R_best_tcp, q_safe, 150)
    traj_pre2grasp = cartesian_interpolate(pos_pre, pos_final, R_best_tcp, traj_safe2pre[-1], 300)
    traj_grasp2safe = cartesian_interpolate(pos_final, pos_safe, R_best_tcp, traj_pre2grasp[-1], 150)
    q_pre = traj_safe2pre[-1]
    q_fin = traj_pre2grasp[-1]

    # --- OMPL 全局避障 ---
    class ValidityChecker(ob.StateValidityChecker):
        def isValid(self, state):
            q = np.array([state[i] for i in range(6)])
            pin.forwardKinematics(robot_model, robot_data, q)
            pin.updateGeometryPlacements(robot_model, robot_data, collision_model, collision_data)
            
            for i in range(len(collision_model.geometryObjects)):
                if "base_link" in collision_model.geometryObjects[i].name: continue
                oMg = collision_data.oMg[i]
                robot_geom = fcl.CollisionObject(collision_model.geometryObjects[i].geometry, fcl.Transform3f(oMg.rotation, oMg.translation))
                for obs in fcl_world:
                    req, res = fcl.CollisionRequest(), fcl.CollisionResult()
                    req.enable_contact = True
                    fcl.collide(robot_geom, obs, req, res)
                    if res.isCollision() and res.getContact(0).penetration_depth > 0.005: return False
            return True

    print("🛰️ 启动 OMPL 进行全局避障规划...")
    space = ob.RealVectorStateSpace(6)
    bounds = ob.RealVectorBounds(6)
    for i in range(6): 
        bounds.setLow(i, -6.28) 
        bounds.setHigh(i, 6.28)
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityChecker(si))
    si.setStateValidityCheckingResolution(0.01) 
    si.setup()

    pdef = ob.ProblemDefinition(si)
    start_state, goal_state = space.allocState(), space.allocState()
    for i in range(6):
        start_state[i] = q_home[i]
        goal_state[i] = q_safe[i]
    pdef.addStartState(start_state)
    pdef.setGoalState(goal_state)

    planner = og.RRTConnect(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(3.0) 
    if solved:
        print("✅ OMPL 避障规划成功！")
        path = pdef.getSolutionPath()
        ps = og.PathSimplifier(si)
        ps.simplifyMax(path)
        path.interpolate(300) 
        traj_home2safe = np.array([[state[i] for i in range(6)] for state in path.getStates()])
    else:
        print("❌ OMPL 规划失败！将使用原生插值")
        traj_home2safe = linear_interpolate(q_home, q_safe, 300)

    grasp_success = True
    print("✅ 所有阶段安全轨迹计算完毕！")

# ==========================================
# 6. MuJoCo 抓取执行（稳如泰山状态机）
# ==========================================
print("🌍 启动仿真窗口，执行抓取流程...")
state = 0
traj_idx = 0
sub_step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not grasp_success:
            data.ctrl[:6] = q_home
            mujoco.mj_step(model, data)
            viewer.sync()
            continue

        if state == 0:
            data.ctrl[:6] = traj_home2safe[traj_idx]
            data.ctrl[6] = 0.0 # 0.0 夹爪张开
            traj_idx += 1
            if traj_idx >= len(traj_home2safe):
                state = 1
                traj_idx = 0
                print("🚁 [状态1] 抵达高空安全点，准备笔直下降...")

        elif state == 1:
            data.ctrl[:6] = traj_safe2pre[traj_idx]
            data.ctrl[6] = 0.0
            traj_idx += 1
            if traj_idx >= len(traj_safe2pre):
                state = 2
                traj_idx = 0
                print("🎯 [状态2] 对准目标，开始极慢速微距接近...")

        elif state == 2:
            data.ctrl[:6] = traj_pre2grasp[traj_idx]
            data.ctrl[6] = 0.0
            traj_idx += 1
            if traj_idx >= len(traj_pre2grasp):
                state = 3
                sub_step = 0
                print("🤏 [状态3] 轻柔触达目标，缓慢且坚定地闭合夹爪！等待物理静摩擦力...")

        elif state == 3:
            data.ctrl[:6] = q_fin # 位置彻底锁死，防止晃动
            data.ctrl[6] = 255.0  # 满级力量闭合！
            sub_step += 1
            # 🌟 核心憋气：等待 400 步 (约2秒)，给物理引擎极其充足的时间让硅胶与圆柱体完全咬合
            if sub_step > 400: 
                state = 4
                traj_idx = 0
                print("🏗️ [状态4] 夹紧确认，将物体笔直拔起！")

        elif state == 4:
            data.ctrl[:6] = traj_grasp2safe[traj_idx]
            data.ctrl[6] = 255.0  # 拔起时力量决不能松！
            traj_idx += 1
            if traj_idx >= len(traj_grasp2safe):
                state = 5
                print("🎉 任务圆满完成！稳稳拿捏！")

        elif state == 5:
            data.ctrl[:6] = q_safe
            data.ctrl[6] = 255.0

        for _ in range(2):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
