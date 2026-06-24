import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from dm_control import mjcf
import random

# ==========================================
# 数学工具：生成锥角范围内的随机四元数
# ==========================================
def random_quat_within_cone(center_quat, max_angle_rad):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, max_angle_rad)
    delta_quat = np.array([
        np.cos(angle/2),
        axis[0] * np.sin(angle/2),
        axis[1] * np.sin(angle/2),
        axis[2] * np.sin(angle/2)
    ])
    res = np.empty(4)
    mujoco.mju_mulQuat(res, center_quat, delta_quat)
    return res

# ==========================================
# MuJoCo原生迭代IK
# ==========================================
def solve_ik_mujoco(model, data, site_name, target_pos, target_rotmat, max_steps=150, tol=1e-3):
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
    
    for attempt in range(4):
        if attempt > 0:
            data.qpos[:6] = best_q + np.random.uniform(-0.2, 0.2, 6)
            if data.qpos[2] < 0:
                data.qpos[2] = -data.qpos[2]
        
        for step in range(max_steps):
            mujoco.mj_kinematics(model, data)
            
            dx = target_pos - data.site_xpos[site_id]
            curr_quat = np.empty(4)
            mujoco.mju_mat2Quat(curr_quat, data.site_xmat[site_id].flatten())
            dr = np.empty(3)
            mujoco.mju_subQuat(dr, target_quat, curr_quat)
            
            err = np.hstack([dx, dr])
            err_norm = np.linalg.norm(err)
            
            if err_norm < min_err:
                min_err = err_norm
                best_q = data.qpos[:6].copy()
            
            if err_norm < tol:
                break
            
            mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)
            jac = np.vstack([jac_p[:, :6], jac_r[:, :6]])
            v = jac.T @ np.linalg.inv(jac @ jac.T + damp * np.eye(6)) @ err
            data.qpos[:6] += step_size * v
        
        if err_norm < tol:
            break
    
    q_res = best_q.copy()
    data.qpos[:] = q_backup
    mujoco.mj_kinematics(model, data)
    return q_res

# ==========================================
# 最终修复版：全姿态柔顺抓取环境
# ==========================================
class ResidualGraspEnv(gym.Env):
    def __init__(self):
        super(ResidualGraspEnv, self).__init__()

        # ========== 1. 创建世界基础场景 ==========
        self.world = mjcf.RootElement()
        self.world.worldbody.add('light', pos=[0, 0, 5], dir=[0, 0, -1], directional=True)
        self.world.worldbody.add('geom', name='floor', type='plane', size=[2, 2, 0.1], rgba=[0.9, 0.9, 0.9, 1])
        table = self.world.worldbody.add('body', name='workbench', pos=[0, 0, 0.2])
        table.add('geom', type='box', size=[0.6, 0.35, 0.02], rgba=[0.6, 0.55, 0.45, 1])

        # ========== 2. 加载模型 + 彻底清理冲突元素 ==========
        self.ur5e = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/scene.xml")
        self.gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/scene.xml")

        # 🔧 核心修复1：删除所有模型自带灯光，彻底消除azimuth冲突
        for light in self.ur5e.find_all('light'):
            light.remove()
        for light in self.gripper.find_all('light'):
            light.remove()

        # 🔧 核心修复2：删除夹爪自带的地面，避免重叠
        for geom in self.gripper.find_all('geom'):
            if geom.type == 'plane':
                geom.remove()

        # ========== 3. 正确挂载（和原版写法完全一致） ==========
        # 先把夹爪挂到机械臂末端
        self.ur5e.find('site', 'attachment_site').attach(self.gripper)
        # 🔧 核心修复3：正确挂载机械臂到世界（RootElement.attach 而非 worldbody.attach）
        self.world.attach(self.ur5e)

        # ========== 4. 添加TCP位点 ==========
        wrist = self.ur5e.find('body', 'wrist_3_link')
        self.tcp_site = wrist.add('site', name='tcp_site', pos=[0, 0, 0.08],
                                  size=[0.005, 0.005, 0.005], rgba=[1, 0, 0, 0.5])

        # ========== 5. 添加抓取物体 ==========
        self.target_body = self.world.worldbody.add('body', name='target_cylinder', pos=[0.25, 0, 0.45])
        self.target_body.add('joint', name='target_freejoint', type='free')
        self.target_body.add('geom', name='cylinder_geom', type='cylinder',
                             size=[0.03, 0.05], mass=0.5, condim=4,
                             friction=[1.0, 0.1, 0.02], rgba=[0.2, 0.5, 0.8, 1])

        # ========== 6. 编译模型 + 驱动器参数 ==========
        self.model = mujoco.MjModel.from_xml_string(self.world.to_xml_string())
        self.data = mujoco.MjData(self.model)

        for i in range(6):
            self.model.actuator_forcelimited[i] = True
            self.model.actuator_forcerange[i] = [-1000.0, 1000.0]
        self.model.actuator_gainprm[6, 0] = 255.0

        # ========== 7. 全姿态参数 ==========
        self.grasp_radius = 0.03
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_site')
        self.target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'target_freejoint')

        self.current_step = 0
        self.last_action = np.zeros(7)
        self.last_obj_z = 0.45

        self.target_grasp_pos = np.zeros(3)
        self.target_grasp_quat = np.zeros(4)
        self.approach_dir = np.zeros(3)

    def _is_gripper_contact_object(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_name = self.model.geom_id2name(con.geom1) or ""
            geom2_name = self.model.geom_id2name(con.geom2) or ""

            obj_in = 'cylinder_geom' in geom1_name or 'cylinder_geom' in geom2_name
            gripper_in = ('finger' in geom1_name or 'knuckle' in geom1_name or
                          'finger' in geom2_name or 'knuckle' in geom2_name)
            if obj_in and gripper_in:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._pre_grasp_rewarded = False
        if hasattr(self, '_contact_step'):
            delattr(self, '_contact_step')

        # 1. 随机物体位置
        target_x = 0.25 + random.uniform(-0.04, 0.04)
        target_y = 0.0 + random.uniform(-0.04, 0.04)
        obj_pos = np.array([target_x, target_y, 0.45])

        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        self.data.qpos[qpos_idx:qpos_idx+3] = obj_pos
        self.data.qpos[qpos_idx+3:qpos_idx+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        # 2. 随机目标抓取姿态
        R_down = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
        base_quat = np.empty(4)
        mujoco.mju_mat2Quat(base_quat, R_down.flatten())

        max_angle = np.deg2rad(10)
        self.target_grasp_quat = random_quat_within_cone(base_quat, max_angle)

        target_grasp_rotmat = np.empty(9)
        mujoco.mju_quat2Mat(target_grasp_rotmat, self.target_grasp_quat)
        target_grasp_rotmat = target_grasp_rotmat.reshape(3,3)

        self.approach_dir = -target_grasp_rotmat[:, 2]

        # 3. 计算目标TCP抓取点
        self.target_grasp_pos = obj_pos + self.approach_dir * self.grasp_radius

        # 4. 预抓取点
        pre_grasp_pos = self.target_grasp_pos + self.approach_dir * 0.02
        pre_grasp_pos += np.random.uniform(-0.015, 0.015, 3)

        q_init = solve_ik_mujoco(self.model, self.data, 'tcp_site', pre_grasp_pos, target_grasp_rotmat)

        self.data.qpos[:6] = q_init
        self.data.ctrl[:6] = q_init
        self.data.ctrl[6] = 0.0
        self.last_action = np.zeros(7)
        self.last_obj_z = 0.45

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        tcp_pos = self.data.site_xpos[self.target_site_id].copy()
        tcp_mat = self.data.site_xmat[self.target_site_id].reshape(3,3).copy()
        tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())

        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        obj_pos = self.data.qpos[qpos_idx:qpos_idx+3].copy()
        vec_to_obj = obj_pos - tcp_pos

        gripper_pos = np.array([self.data.ctrl[6]])
        joint_forces = self.data.qfrc_actuator[:6].copy() * 0.01

        obs = np.concatenate([
            tcp_pos, tcp_quat,
            obj_pos, vec_to_obj,
            gripper_pos, joint_forces,
            self.target_grasp_quat
        ])
        return obs.astype(np.float32)

    def _compute_reward_and_done(self, obs, action):
        tcp_pos = obs[:3]
        tcp_quat = obs[3:7]
        gripper_pos = obs[13]

        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        obj_pos = self.data.qpos[qpos_idx:qpos_idx+3].copy()
        obj_z = obj_pos[2]

        reward = 0.0
        done = False
        is_success = False

        if obj_z < 0.415:
            reward -= 100.0
            done = True
            return reward, done, is_success

        obj_quat = self.data.qpos[qpos_idx+3:qpos_idx+7]
        body_z = np.array([0, 0, 1])
        obj_z_axis = np.empty(3)
        mujoco.mju_rotVecQuat(obj_z_axis, body_z, obj_quat)
        tilt_angle = np.arccos(np.clip(obj_z_axis[2], -1, 1))
        if tilt_angle > np.deg2rad(45):
            reward -= 50.0
            done = True
            return reward, done, is_success

        # 位姿对齐奖励
        pos_err = np.linalg.norm(tcp_pos - self.target_grasp_pos)
        reward += np.clip(0.04 - pos_err, 0, 0.04) * 50

        rot_err = 1.0 - np.abs(np.dot(tcp_quat, self.target_grasp_quat))
        reward += np.clip(0.2 - rot_err, 0, 0.2) * 20

        if pos_err < 0.015 and rot_err < 0.08 and not self._pre_grasp_rewarded:
            reward += 10.0
            self._pre_grasp_rewarded = True

        # 接触与夹持奖励
        has_contact = self._is_gripper_contact_object()
        if has_contact:
            reward += 2.0
            reward += gripper_pos * 5.0
        else:
            reward -= gripper_pos * 2.0

        # 拔起成功奖励
        lift_height = obj_z - 0.45
        if lift_height > 0.003 and has_contact and gripper_pos > 0.6:
            reward += 200.0
            is_success = True
            done = True

        if lift_height > 0:
            reward += lift_height * 2000

        # 正则项
        action_diff = np.linalg.norm(action - self.last_action)
        reward -= 0.05 * action_diff
        self.last_action = action.copy()

        reward -= 0.3
        return reward, done, is_success

    def step(self, action):
        self.current_step += 1

        dx = action[:3] * 0.015
        dr = action[3:6] * 0.025
        gripper_delta = action[6] * 0.08

        curr_pos = self.data.site_xpos[self.target_site_id].copy()
        target_pos = curr_pos + dx

        target_pos[0] = np.clip(target_pos[0], 0.15, 0.35)
        target_pos[1] = np.clip(target_pos[1], -0.15, 0.15)
        target_pos[2] = np.clip(target_pos[2], 0.42, 0.6)

        dx = target_pos - curr_pos

        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_p, jac_r, self.target_site_id)

        J = np.vstack([jac_p[:, :6], jac_r[:, :6]])
        err = np.hstack([dx, dr])
        dq = J.T @ np.linalg.inv(J @ J.T + 5e-3 * np.eye(6)) @ err

        self.data.ctrl[:6] += dq
        self.data.ctrl[6] = np.clip(self.data.ctrl[6] + gripper_delta, 0.0, 1.0)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done, is_success = self._compute_reward_and_done(obs, action)

        truncated = self.current_step >= 200
        if truncated:
            done = True

        info = {"is_success": is_success}
        return obs, reward, done, truncated, info

if __name__ == "__main__":
    print("🚀 正在初始化全姿态抓取环境...")
    env = ResidualGraspEnv()
    obs, info = env.reset()
    print("✅ 环境初始化完成！")
    print(f"观测维度: {obs.shape}  (预期24)")
    print(f"动作空间: {env.action_space.shape}  (预期7)")
    print(f"目标抓取点位置: {env.target_grasp_pos}")
    print(f"接近方向: {env.approach_dir}")
