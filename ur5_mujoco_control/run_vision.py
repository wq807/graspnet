import os
import sys
import time
import random
import zmq
import numpy as np
import pinocchio as pin
import hppfcl as fcl
import mujoco
import mujoco.viewer
from dm_control import mjcf
from scipy.optimize import minimize

# ==========================================
# 🌟 原生加载 SB3 与你的训练环境
# ==========================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from residual_grasp_env_3dof import ResidualGraspEnv  

# ==========================================
# OMPL 与底层 IK 工具
# ==========================================
sys.path.insert(0, '/home/wq/ur_graspnet/ur5_mujoco_control/ompl/build/py-bindings')
from ompl._ompl import base as ob
from ompl._ompl import geometric as og

def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi

def linear_interpolate(q_start, q_end, steps=300):
    t = np.linspace(0, 1, steps)
    return np.array([q_start + (q_end - q_start) * ti for ti in t])
    
def solve_ik_scipy(model, data, site_name, target_pos, target_rotmat, q_seed=None):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, target_rotmat.flatten())
    q_backup = data.qpos[:6].copy()
    
    def objective(q):
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        curr_pos = data.site_xpos[site_id]
        curr_quat = np.empty(4)
        mujoco.mju_mat2Quat(curr_quat, data.site_xmat[site_id].flatten())
        pos_err = np.sum((curr_pos - target_pos)**2)
        rot_err = 1.0 - np.abs(np.dot(curr_quat, target_quat))
        return pos_err + 0.5 * rot_err

    bounds = [(-2*np.pi, 2*np.pi)] * 6
    bounds[2] = (0.1, np.pi) 
    
    seeds = [[0.0, -1.57, 1.57, -1.57, -1.57, 0.0]]
    if q_seed is not None:
        seeds.insert(0, q_seed)
        
    best_res = None
    best_val = float('inf')
    for seed in seeds:
        res = minimize(objective, seed, bounds=bounds, method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
        if best_val < 0.001: break
            
    data.qpos[:6] = q_backup
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return best_res.x

def cartesian_interpolate(model, data, pos_start, pos_end, rot_target, q_start, steps=100):
    traj, q_curr = [], q_start.copy()
    for i in range(steps):
        p = pos_start + (pos_end - pos_start) * (i / max(1, steps - 1))
        q_curr = solve_ik_scipy(model, data, 'tcp_site', p, rot_target, q_seed=q_curr)
        traj.append(q_curr.copy())
    return np.array(traj)

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

for geom in gripper.find_all('geom'):
    if geom.name and ('finger' in geom.name.lower() or 'pad' in geom.name.lower() or 'left' in geom.name.lower() or 'right' in geom.name.lower()):
        geom.friction = [1.5, 0.2, 0.001]
        geom.condim = 4

arm_mount = world.worldbody.add('site', name='arm_mount', pos=[-0.25, 0, 0.4])
arm_mount.attach(ur5e)

wrist = ur5e.find('body', 'wrist_3_link')
if wrist is None: wrist = ur5e.find('body', 'ur5e_wrist_3_link')
tcp_site = wrist.add('site', name='tcp_site', pos=[0, 0, 0.08], size=[0.005], rgba=[1, 0, 0, 1])

target_x = random.uniform(0.21, 0.29)
target_y = random.uniform(-0.04, 0.04)
target_body = world.worldbody.add('body', name='target_cylinder', pos=[target_x, target_y, 0.45])
target_body.add('joint', name='target_joint', type='free')
target_body.add('geom', name='cylinder_geom', type='cylinder', 
                 size=[0.03, 0.05], rgba=[0.2, 0.5, 0.8, 1], mass=0.5, 
                 condim=4, friction=[1.0, 0.1, 0.0001])

world.worldbody.add('camera', name='workspace_cam', pos=[0.8, 0, 0.95], fovy=45, xyaxes=[0, 1, 0, -0.707, 0, 0.707])

# ==========================================
# 2. 物理引擎预热与 PPO 装载
# ==========================================
print("⏳ [2/5] 物理引擎预热...")
model = mujoco.MjModel.from_xml_string(world.to_xml_string(), world.get_assets())

for i in range(6):
    model.actuator_forcerange[i] = [-1000.0, 1000.0]
    if model.actuator_gainprm[i, 0] < 500.0:
        model.actuator_gainprm[i, 0] = 500.0
model.actuator_forcerange[6] = [-200.0, 200.0]  
model.actuator_gainprm[6, 0] = 100.0            
data = mujoco.MjData(model)
home_pose = [3.14, -1.57, 1.57, -1.57, -1.57, 0.0]

data.ctrl[:6] = home_pose
data.ctrl[6] = 0.0 
for _ in range(500): mujoco.mj_step(model, data)

print("🔧 [PPO] 正在装载柔顺微操小脑 (注入 VecNormalize 视觉层)...")
dummy_env = DummyVecEnv([lambda: ResidualGraspEnv()])
norm_path = "./logs_residual/vec_normalize.pkl"
vec_env = VecNormalize.load(norm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

fine_tune_args = {
    "learning_rate": 2e-5,
    "ent_coef": 0.03,
    "n_epochs": 5,
    "clip_range": 0.12
}

ppo_brain = PPO.load("./logs_residual/latest_model.zip", custom_objects=fine_tune_args)
print("✅ [PPO] 小脑装载完毕，思维已完全对齐，随时准备接管！")

# ==========================================
# 3. 视觉采集与点云生成
# ==========================================
print("📸 [3/5] 正在拍照...")
width, height = 640, 480
renderer = mujoco.Renderer(model, height=height, width=width)
renderer.update_scene(data, camera='workspace_cam')
renderer.enable_depth_rendering()
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
idx = np.random.choice(len(pc_cv_final), num_points, replace=(len(pc_cv_final) < num_points))
pc_sample = pc_cv_final[idx].astype(np.float32)

# ==========================================
# 4. GraspNet推理
# ==========================================
print("📡 [4/5] 发送纯 JSON 数据至 GraspNet 大脑...")
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
payload = {
    "point_cloud": pc_sample.tolist(),
    "R_cv2world": R_cv2world.tolist()
}
socket.send_json(payload)
best_grasp = socket.recv_json()
socket.close()
context.term()

q_home = np.array(home_pose)
grasp_success = False

if best_grasp["success"]:
    print(f"🎉 [5/5] 收到 GraspNet 战略定位！分数: {best_grasp['score']:.4f}")
    
    grasp_center_cam = np.array(best_grasp["translation"])
    grasp_center_world = R_cv2world @ grasp_center_cam + cam_pos
    
    # 🌟 修复 1：视觉光学偏差补偿！往回推 0.02 米
    grasp_center_world[0] -= 0.02
    
    R_train_down = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    R_best_tcp = R_train_down
    
    pos_pre = np.array([grasp_center_world[0], grasp_center_world[1], 0.47])  
    pos_safe = pos_pre.copy()
    pos_safe[2] += 0.20

    print("🏗️ 正在求解 OMPL 全局宏观航线...")
    URDF_PATH = "ur5e.urdf"
    PKG_DIR = os.path.abspath("mujoco_menagerie")
    robot_model = pin.buildModelFromUrdf(URDF_PATH)
    robot_data = robot_model.createData()
    collision_model = pin.buildGeomFromUrdf(robot_model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs=[PKG_DIR])
    collision_data = collision_model.createData()

    fcl_world = [fcl.CollisionObject(fcl.Box(1.3, 0.8, 0.42), fcl.Transform3f(np.eye(3), np.array([0, 0, 0.21])))]

    def get_ik_mujoco(target_pos_world, target_rot_world, q_seed):
        q_bak = data.qpos[:6].copy()
        data.qpos[:6] = q_seed.copy()
        mujoco.mj_kinematics(model, data)
        q_res = solve_ik(model, data, 'tcp_site', target_pos_world, target_rot_world)
        data.qpos[:6] = q_bak
        mujoco.mj_kinematics(model, data)
        return q_res

    q_safe = solve_ik_scipy(model, data, 'tcp_site', pos_safe, R_best_tcp, q_seed=q_home)
    traj_safe2pre = cartesian_interpolate(model, data, pos_safe, pos_pre, R_best_tcp, q_safe, 150)
    
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
        print("✅ OMPL 避障规划成功！安全抵达目标上空！")
        path = pdef.getSolutionPath()
        ps = og.PathSimplifier(si)
        ps.simplifyMax(path)
        path.interpolate(300) 
        traj_home2safe = np.array([[state[i] for i in range(6)] for state in path.getStates()])
    else:
        print("❌ OMPL 规划失败！启用安全环绕插值。")
        q_safe_unrolled = q_home + wrap_to_pi(q_safe - q_home)
        traj_home2safe = linear_interpolate(q_home, q_safe_unrolled, 300)
    grasp_success = True

# ==========================================
# 5. 双脑联合状态机
# ==========================================
print("🌍 启动仿真窗口，双脑系统上线...")
tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_site')
target_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_joint')

# 🌟 初始化回城状态机所需的所有变量
state = 0
traj_idx = 0
ppo_steps = 0
ppo_max_steps = 200
q_lift_start = None
traj_lift = None
lift_idx = 0
traj_return_home = None 

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not grasp_success:
            data.ctrl[:6] = q_home
            mujoco.mj_step(model, data)
            viewer.sync()
            continue

        if state == 0:
            data.ctrl[:6] = traj_home2safe[traj_idx]
            data.ctrl[6] = 0.0 
            traj_idx += 1
            if traj_idx >= len(traj_home2safe):
                state = 1
                traj_idx = 0
                print("🚁 [GraspNet 大局观] 抵达安全点，飞向预抓取点...")

        elif state == 1:
            data.ctrl[:6] = traj_safe2pre[traj_idx]
            data.ctrl[6] = 0.0
            traj_idx += 1
            if traj_idx >= len(traj_safe2pre):
                print("⏳ 等待机械臂物理沉降 (彻底对齐训练环境的起点)...")
                for _ in range(200):  
                    mujoco.mj_step(model, data)
                viewer.sync()
                
                state = 2
                ppo_steps = 0
                data.ctrl[6] = 0.0  
                print("🦅 [模式切换] 完美对位！PPO 微操小脑已睁开眼睛！")

        elif state == 2:
            tcp_pos = data.site_xpos[tcp_site_id].copy()
            tcp_mat = data.site_xmat[tcp_site_id].copy().reshape(3, 3)
            tcp_quat = np.empty(4)
            mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
            
            qpos_idx = model.jnt_qposadr[target_joint_id]
            target_pos = data.qpos[qpos_idx : qpos_idx+3].copy()
            vec_to_target = target_pos - tcp_pos
            gripper_pos = np.array([data.ctrl[6]])
            joint_forces = data.qfrc_actuator[:6].copy() * 0.01  
            
            obs_raw = np.concatenate([tcp_pos, tcp_quat, target_pos, vec_to_target, gripper_pos, joint_forces]).astype(np.float32)
            
            obs_batched = np.array([obs_raw])
            obs_norm = vec_env.normalize_obs(obs_batched)[0]
            action, _ = ppo_brain.predict(obs_norm, deterministic=True)

            dx = action[:3] * 0.015  
            dr = np.zeros(3)  
            
            current_tcp_pos = data.site_xpos[tcp_site_id].copy()
            target_tcp_pos = current_tcp_pos + dx
            
            target_tcp_pos[0] = np.clip(target_tcp_pos[0], 0.15, 0.35) 
            target_tcp_pos[1] = np.clip(target_tcp_pos[1], -0.15, 0.15)
            target_tcp_pos[2] = np.clip(target_tcp_pos[2], 0.42, 0.6) 
            
            dx = target_tcp_pos - current_tcp_pos
            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac_p, jac_r, tcp_site_id)
            
            J = np.vstack([jac_p, jac_r])[:, :6]
            err = np.hstack([dx, dr])
            dq = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6)) @ err
            q_target = data.ctrl[:6] + dq
            data.ctrl[:6] = q_target

            gripper_delta = action[3] * 0.08
            current_gripper = data.ctrl[6]
            data.ctrl[6] = np.clip(current_gripper + gripper_delta, 0.0, 1.0)

            ppo_steps += 1
            
            # 🌟 修复触发条件，彻底抛弃致命的 Jacobian 抬升，改用安全的无奇异点插值抬升！
            target_z = target_pos[2]
            if target_z > 0.455 and data.ctrl[6] > 0.3:
                state = 3
                q_lift_start = data.qpos[:6].copy()
                # 算出一条从当前抓取点到空中安全点（OMPL航线的尽头）的无缝直连插值
                traj_lift = linear_interpolate(q_lift_start, traj_home2safe[-1], 150)
                lift_idx = 0
                print(f"🎉 [PPO 捷报] 完美锁死！PPO 仅用时 {ppo_steps} 步！移交大局观抬升...")
            elif ppo_steps >= ppo_max_steps:
                state = 3
                q_lift_start = data.qpos[:6].copy()
                traj_lift = linear_interpolate(q_lift_start, traj_home2safe[-1], 150)
                lift_idx = 0
                print("⚠️ [PPO 警告] 达到最大微操步数，强制执行抬起收网。")

        elif state == 3:
            current_gripper = data.ctrl[6]
            if current_gripper < 0.95:
                # 给一点时间让它死死咬紧，夹持期间关节锁死不抖动
                data.ctrl[6] = np.clip(current_gripper + 0.05, 0.0, 1.0)
                data.ctrl[:6] = q_lift_start 
            else:
                data.ctrl[6] = 1.0 
                # 🚀 执行无奇异点的绝对安全抬升！
                data.ctrl[:6] = traj_lift[lift_idx]
                lift_idx += 1
                
                # 到达高空安全点后，触发回城
                if lift_idx >= len(traj_lift):
                    state = 4
                    traj_idx = 0
                    # 直接把当初飞过来的安全航线倒放一遍，原路退回老家！绝不碰桌子！
                    traj_return_home = traj_home2safe[::-1]
                    print("🏆 抓取微操完毕！正在沿 OMPL 安全航线原路返回老家...")

        # 🏠 执行安全回城录像倒放
        elif state == 4:
            data.ctrl[:6] = traj_return_home[traj_idx]
            data.ctrl[6] = 1.0  
            traj_idx += 1
            if traj_idx >= len(traj_return_home):
                state = 5
                print("🏠 已安全带着战利品回到老家！任务圆满结束！")

        # ⏹️ 永久定格
        elif state == 5:
            data.ctrl[:6] = q_home
            data.ctrl[6] = 1.0 

        steps_to_run = 10 if state == 2 else 2
        for _ in range(steps_to_run):
            mujoco.mj_step(model, data)
            
        viewer.sync()
        time.sleep(0.005)
