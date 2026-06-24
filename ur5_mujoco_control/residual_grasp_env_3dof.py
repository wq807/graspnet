import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from dm_control import mjcf
import random
from scipy.optimize import minimize

# ==========================================
# 数学与物理底层工具
# ==========================================
def solve_ik_scipy(model, data, site_name, target_pos, target_rotmat):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, target_rotmat.flatten())
    q_backup = data.qpos.copy()
    
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
    seeds = [
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        [0.5, -2.0, 2.0, -1.57, -1.57, 0.0],
        [-0.5, -1.0, 1.0, -1.57, -1.57, 0.0]
    ]
    
    best_res = None
    best_val = float('inf')
    for seed in seeds:
        res = minimize(objective, seed, bounds=bounds, method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
        if best_val < 0.001: break
            
    data.qpos[:] = q_backup
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return best_res.x

# ==========================================
# 柔顺强化学习核心环境
# ==========================================
class ResidualGraspEnv(gym.Env):
    def __init__(self):
        super(ResidualGraspEnv, self).__init__()
        
        self.world = mjcf.RootElement()
        self.world.worldbody.add('light', pos=[0, 0, 5], dir=[0, 0, -1], directional=True)
        self.world.worldbody.add('geom', name='floor', type='plane', size=[2, 2, 0.1], rgba=[0.9, 0.9, 0.9, 1])
        table = self.world.worldbody.add('body', name='workbench', pos=[0, 0, 0.2])
        table.add('geom', type='box', size=[0.6, 0.35, 0.2], rgba=[0.65, 0.55, 0.45, 1])

        # 加载机器人与夹爪模型
        self.ur5e = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
        self.gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")
        self.ur5e.find('site', 'attachment_site').attach(self.gripper)

        # 配置夹爪高摩擦，解决打滑
        gripper_geoms = self.gripper.find_all('geom')
        for geom in gripper_geoms:
            if geom.name is None:
                continue
            name_lower = geom.name.lower()
            if 'finger' in name_lower or 'pad' in name_lower or 'left' in name_lower or 'right' in name_lower:
                geom.friction = [1.5, 0.2, 0.001]
                geom.condim = 4

        arm_mount = self.world.worldbody.add('site', name='arm_mount', pos=[-0.25, 0, 0.4])
        arm_mount.attach(self.ur5e)
        
        wrist = self.ur5e.find('body', 'wrist_3_link')
        if wrist is None: wrist = self.ur5e.find('body', 'ur5e_wrist_3_link')
        self.tcp_site = wrist.add('site', name='tcp_site', pos=[0, 0, 0.08], size=[0.005], rgba=[1, 0, 0, 1])

        # 目标物体
        self.target_body = self.world.worldbody.add('body', name='target_cylinder', pos=[0.25, 0, 0.45])
        self.target_body.add('joint', name='target_joint', type='free')
        self.target_body.add('geom', name='cylinder_geom', type='cylinder', 
                             size=[0.03, 0.05], mass=0.5, condim=4, 
                             friction=[1.0, 0.1, 0.0001])

        self.model = mujoco.MjModel.from_xml_string(self.world.to_xml_string(), self.world.get_assets())
        
        # 执行器参数配置
        for i in range(6):
            self.model.actuator_forcerange[i] = [-1000.0, 1000.0]
            if self.model.actuator_gainprm[i, 0] < 500.0:
                self.model.actuator_gainprm[i, 0] = 500.0
        
        self.model.actuator_forcerange[6] = [-3000.0, 3000.0]
        self.model.actuator_gainprm[6, 0] = 1500.0
                
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_site')
        self.target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'target_joint')
        self.current_step = 0
        self.last_action = np.zeros(4)
        self.last_obj_z = 0.45

    def _is_gripper_contact_object(self):
        """检测夹爪指尖是否与目标圆柱体产生物理接触"""
        obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cylinder_geom')
        if obj_geom_id == -1:
            return False
        
        gripper_keywords = ['finger', 'pad', 'left', 'right']
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            
            has_object = (g1 == obj_geom_id) or (g2 == obj_geom_id)
            if not has_object:
                continue
            
            other_geom = g2 if g1 == obj_geom_id else g1
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            if geom_name is None:
                continue
            
            if any(kw in geom_name.lower() for kw in gripper_keywords):
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 重置回合内状态标记
        self._pre_grasp_rewarded = False
        if hasattr(self, '_contact_step'):
            delattr(self, '_contact_step')

        # 扩大物体随机范围，提升泛化性
        target_x = 0.25 + random.uniform(-0.04, 0.04)
        target_y = 0.0  + random.uniform(-0.04, 0.04)
        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        self.data.qpos[qpos_idx : qpos_idx+3] = [target_x, target_y, 0.45]
        self.data.qpos[qpos_idx+3 : qpos_idx+7] = [1, 0, 0, 0] 
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # 预抓取位姿增加扰动，避免开局贴脸
        pre_grasp_pos = np.array([
            target_x + random.uniform(-0.02, 0.02),
            target_y + random.uniform(-0.02, 0.02),
            0.47
        ])
        
        R_down = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
        q_init = solve_ik_scipy(self.model, self.data, 'tcp_site', pre_grasp_pos, R_down)
        
        self.data.qpos[:6] = q_init
        self.data.ctrl[:6] = q_init
        self.data.ctrl[6] = 0.0 
        self.last_action = np.zeros(4)
        self.last_obj_z = 0.45
        
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
            
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        tcp_pos = self.data.site_xpos[self.target_site_id].copy()
        tcp_mat = self.data.site_xmat[self.target_site_id].copy().reshape(3, 3)
        tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
        
        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        target_pos = self.data.qpos[qpos_idx : qpos_idx+3].copy()
        vec_to_target = target_pos - tcp_pos
        gripper_pos = np.array([self.data.ctrl[6]])
        joint_forces = self.data.qfrc_actuator[:6].copy() * 0.01
        
        obs = np.concatenate([tcp_pos, tcp_quat, target_pos, vec_to_target, gripper_pos, joint_forces])
        return obs.astype(np.float32)

    def _compute_reward_and_done(self, obs, action):
        target_pos = obs[7:10]
        vec_to_target = obs[10:13]
        gripper_pos = obs[13] 
        
        xy_dist = np.linalg.norm(vec_to_target[:2])
        tcp_z = obs[2]
        target_z = target_pos[2]
        reward = 0.0
        done = False
        is_success = False

        # ===== 1. 失败检测与惩罚 =====
        qpos_idx = self.model.jnt_qposadr[self.target_joint_id]
        target_quat = self.data.qpos[qpos_idx+3 : qpos_idx+7].copy()
        rot_mat = np.empty(9)
        mujoco.mju_quat2Mat(rot_mat, target_quat)
        body_z = rot_mat[2::3]
        tilt_angle = np.arccos(np.clip(np.dot(body_z, [0,0,1]), -1, 1))
        
        if tilt_angle > 0.785 or target_z < 0.415:
            reward -= 100.0
            done = True
            return reward, done, is_success
        
        if tilt_angle > 0.26:
            reward -= (tilt_angle - 0.26) * 50

        # ===== 2. 位置对齐：低权重引导 =====
        reward += np.clip(0.04 - xy_dist, 0, 0.04) * 3
        ideal_grasp_z = 0.45
        z_err = abs(tcp_z - ideal_grasp_z)
        reward += np.clip(0.05 - z_err, 0, 0.05) * 2

        # 预抓取位姿：一次性奖励，不重复发放
        if xy_dist < 0.02 and z_err < 0.02 and not self._pre_grasp_rewarded:
            reward += 2.0
            self._pre_grasp_rewarded = True

        # ===== 3. 接触与夹持：限时奖励，防止刷分 =====
        is_contact = self._is_gripper_contact_object()
        
        if is_contact:
            # 首次接触记录步数
            if not hasattr(self, '_contact_step'):
                self._contact_step = self.current_step
            contact_duration = self.current_step - self._contact_step
            
            # 接触奖励仅前30步有效，超时未拔起逐步惩罚
            if contact_duration < 30:
                reward += gripper_pos * 20.0
            else:
                reward -= (contact_duration - 30) * 0.3
            
            if gripper_pos > 0.5:
                reward += 5.0
        else:
            # 无接触时合爪惩罚
            reward -= gripper_pos * 2.0
            # 失去接触重置计时器
            if hasattr(self, '_contact_step'):
                delattr(self, '_contact_step')

        # ===== 4. 拔起奖励：核心收益，大幅提权 =====
        lift_height = target_z - 0.45
        if lift_height > 0 and gripper_pos > 0.2:
            reward += lift_height * (gripper_pos ** 2) * 8000
            
            # 阶梯里程碑奖励
            if self.last_obj_z <= 0.451 and target_z > 0.451:
                reward += 80.0
            if self.last_obj_z <= 0.453 and target_z > 0.453:
                reward += 300.0
        
        self.last_obj_z = target_z

        # ===== 5. 成功终态大奖 =====
        if target_z > 0.455 and gripper_pos > 0.3 and tilt_angle < 0.35:
            reward += 800.0
            is_success = True
            done = True

        # ===== 6. 正则项与步数惩罚 =====
        action_diff = np.linalg.norm(action - self.last_action)
        reward -= 0.05 * action_diff
        self.last_action = action.copy()

        reward -= 0.5

        return reward, done, is_success

    def step(self, action):
        self.current_step += 1
        dx = action[:3] * 0.015  
        dr = np.zeros(3) 
        current_tcp_pos = self.data.site_xpos[self.target_site_id].copy()
        target_tcp_pos = current_tcp_pos + dx
        
        target_tcp_pos[0] = np.clip(target_tcp_pos[0], 0.15, 0.35) 
        target_tcp_pos[1] = np.clip(target_tcp_pos[1], -0.15, 0.15)
        target_tcp_pos[2] = np.clip(target_tcp_pos[2], 0.42, 0.6) 
        
        dx = target_tcp_pos - current_tcp_pos
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_p, jac_r, self.target_site_id)
        
        J = np.vstack([jac_p, jac_r])[:, :6]
        err = np.hstack([dx, dr])
        dq = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6)) @ err
        q_target = self.data.ctrl[:6] + dq
        self.data.ctrl[:6] = q_target

        gripper_delta = action[3] * 0.08
        current_gripper = self.data.ctrl[6]
        self.data.ctrl[6] = np.clip(current_gripper + gripper_delta, 0.0, 1.0)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done, is_success = self._compute_reward_and_done(obs, action)
        truncated = self.current_step >= 200
        info = {"is_success": is_success}
        
        return obs, reward, done, truncated, info

if __name__ == "__main__":
    print("🚀 正在初始化残差柔顺抓取环境...")
    env = ResidualGraspEnv()
    obs, info = env.reset()
    print("✅ 环境初始化完成！")
    print(f"观测维度: {obs.shape}")
    print(f"动作空间: {env.action_space.shape}")
