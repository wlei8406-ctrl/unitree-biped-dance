import os
import tempfile
from typing import Optional
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt


class UnitreeBipedDanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.physics_client = None
        self.robot_id = None
        self.num_joints = 8
        self.joint_indices = None
        self.t = 0.0
        self.step_count = 0
        self.max_steps = 1500

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(8 + 8 + 9 + 8,), dtype=np.float32
        )

    def _build_urdf(self):
        urdf = '''<?xml version="1.0"?>
<robot name="biped">
  <link name="torso">
    <inertial><mass value="5.0"/><inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.06"/></inertial>
    <visual><geometry><box size="0.36 0.24 0.50"/></geometry></visual>
    <collision><geometry><box size="0.36 0.24 0.50"/></geometry></collision>
  </link>
  <link name="left_thigh">
    <inertial><mass value="1.2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/></inertial>
    <visual><origin xyz="0 0 -0.23"/><geometry><box size="0.10 0.10 0.46"/></geometry></visual>
    <collision><origin xyz="0 0 -0.23"/><geometry><box size="0.10 0.10 0.46"/></geometry></collision>
  </link>
  <link name="left_calf">
    <inertial><mass value="1.0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.008"/></inertial>
    <visual><origin xyz="0 0 -0.22"/><geometry><box size="0.09 0.09 0.44"/></geometry></visual>
    <collision><origin xyz="0 0 -0.22"/><geometry><box size="0.09 0.09 0.44"/></geometry></collision>
  </link>
  <link name="left_foot">
    <inertial><mass value="0.3"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0.05 0 0"/><geometry><box size="0.22 0.12 0.08"/></geometry></visual>
    <collision><origin xyz="0.05 0 0"/><geometry><box size="0.22 0.12 0.08"/></geometry></collision>
  </link>
  <link name="right_thigh">
    <inertial><mass value="1.2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/></inertial>
    <visual><origin xyz="0 0 -0.23"/><geometry><box size="0.10 0.10 0.46"/></geometry></visual>
    <collision><origin xyz="0 0 -0.23"/><geometry><box size="0.10 0.10 0.46"/></geometry></collision>
  </link>
  <link name="right_calf">
    <inertial><mass value="1.0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.008"/></inertial>
    <visual><origin xyz="0 0 -0.22"/><geometry><box size="0.09 0.09 0.44"/></geometry></visual>
    <collision><origin xyz="0 0 -0.22"/><geometry><box size="0.09 0.09 0.44"/></geometry></collision>
  </link>
  <link name="right_foot">
    <inertial><mass value="0.3"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0.05 0 0"/><geometry><box size="0.22 0.12 0.08"/></geometry></visual>
    <collision><origin xyz="0.05 0 0"/><geometry><box size="0.22 0.12 0.08"/></geometry></collision>
  </link>
  <joint name="left_hip_yaw" type="revolute"><parent link="torso"/><child link="left_thigh"/><origin xyz="0.12 0.08 -0.24"/><axis xyz="0 0 1"/><limit lower="-0.8" upper="0.8" effort="1000" velocity="10"/></joint>
  <joint name="left_hip_pitch" type="revolute"><parent link="left_thigh"/><child link="left_calf"/><origin xyz="0 0 -0.46"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="0.5" effort="1000" velocity="10"/></joint>
  <joint name="left_knee_pitch" type="revolute"><parent link="left_calf"/><child link="left_foot"/><origin xyz="0 0 -0.44"/><axis xyz="0 1 0"/><limit lower="-2.5" upper="0.5" effort="1000" velocity="10"/></joint>
  <joint name="left_ankle_pitch" type="revolute"><parent link="left_foot"/><child link="torso"/><origin xyz="0.1 0 0"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.0" effort="500" velocity="10"/></joint>
  <joint name="right_hip_yaw" type="revolute"><parent link="torso"/><child link="right_thigh"/><origin xyz="0.12 -0.08 -0.24"/><axis xyz="0 0 1"/><limit lower="-0.8" upper="0.8" effort="1000" velocity="10"/></joint>
  <joint name="right_hip_pitch" type="revolute"><parent link="right_thigh"/><child link="right_calf"/><origin xyz="0 0 -0.46"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="0.5" effort="1000" velocity="10"/></joint>
  <joint name="right_knee_pitch" type="revolute"><parent link="right_calf"/><child link="right_foot"/><origin xyz="0 0 -0.44"/><axis xyz="0 1 0"/><limit lower="-2.5" upper="0.5" effort="1000" velocity="10"/></joint>
  <joint name="right_ankle_pitch" type="revolute"><parent link="right_foot"/><child link="torso"/><origin xyz="0.1 0 0"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.0" effort="500" velocity="10"/></joint>
</robot>
'''
        fd, path = tempfile.mkstemp(suffix=".urdf", text=True)
        os.close(fd)
        with open(path, "w") as f:
            f.write(urdf)
        return path

    def _reset_sim(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 120.0)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_path = self._build_urdf()
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 1.1], useFixedBase=False)
        self.joint_indices = list(range(p.getNumJoints(self.robot_id)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_sim()
        self.t = 0.0
        self.step_count = 0
        init_q = np.array([0.0, -0.7, 0.0, 0.0, 0.0, -0.7, 0.0, 0.0], dtype=np.float32)
        for j, q in zip(self.joint_indices, init_q):
            p.resetJointState(self.robot_id, j, targetValue=q, targetVelocity=0.0)
        for _ in range(40):
            p.stepSimulation()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        js = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        jv = [p.getJointState(self.robot_id, j)[1] for j in self.joint_indices]
        js = np.asarray(js, dtype=np.float32)
        jv = np.asarray(jv, dtype=np.float32)
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rpy = np.asarray(p.getEulerFromQuaternion(orn), dtype=np.float32)
        lin_vel = np.asarray(p.getBaseVelocity(self.robot_id)[0], dtype=np.float32)
        ang_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1], dtype=np.float32)
        target_q = self._dance_target(self.t)
        obs = np.concatenate([js, jv, rpy, lin_vel, ang_vel, target_q])
        return obs.astype(np.float32)

    def _dance_target(self, t):
        phase = 2.5 * t
        return np.array([
            0.25 * np.sin(phase),
            -0.8 - 0.25 * np.sin(phase + 1.3),
            0.4 * np.sin(phase + 2.2),
            0.15 * np.sin(phase + 3.1),
            -0.25 * np.sin(phase),
            -0.8 - 0.25 * np.sin(phase + 2.6),
            0.4 * np.sin(phase + 3.5),
            0.15 * np.sin(phase + 4.0),
        ], dtype=np.float32)

    def _apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        current = np.array([p.getJointState(self.robot_id, j)[0] for j in self.joint_indices])
        target = current + 0.18 * action
        target = np.clip(target, -2.5, 2.5)
        for j, q in zip(self.joint_indices, target):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL,
                targetPosition=q, positionGain=1.0, velocityGain=0.5, force=90.0,
            )

    def step(self, action):
        self._apply_action(action)
        for _ in range(10):
            p.stepSimulation()
        self.t += 1.0 / 120.0
        self.step_count += 1
        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs)
        truncated = self.step_count >= self.max_steps
        return obs, float(reward), terminated or truncated, truncated, {"step": self.step_count}

    def _compute_reward(self, obs):
        joint_states = obs[: self.num_joints]
        joint_vels = obs[self.num_joints: self.num_joints * 2]
        target_q = obs[self.num_joints * 2 + 9:]
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        base_height = pos[2]
        height_reward = -abs(base_height - 1.1) * 8.0
        posture_reward = -(abs(roll) + abs(pitch)) * 4.0
        tracking_err = np.abs(joint_states - target_q)
        tracking_reward = -np.mean(tracking_err) * 6.0
        smooth_reward = -0.15 * np.mean(np.abs(joint_vels))
        fall_penalty = -30.0 if (base_height < 0.5 or abs(roll) > 1.4 or abs(pitch) > 1.4) else 0.0
        reward = height_reward + posture_reward + tracking_reward + smooth_reward + fall_penalty
        terminated = base_height < 0.4
        return reward, terminated

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 双足机器人舞蹈强化学习训练 (PPO + PyBullet)")
    print("="*80 + "\n")
    
    print("[1/3] 创建训练环境...")
    env = DummyVecEnv([lambda: UnitreeBipedDanceEnv(render_mode=None)])
    
    model_path = "unitree_biped_ppo"
    
    print("[2/3] 初始化 PPO 模型...")
    if os.path.exists(model_path + ".zip"):
        print("✓ 发现已有模型，继续训练...\n")
        model = PPO.load(model_path, env=env)
    else:
        print("✓ 新建模型，从零开始训练...\n")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
        )
    
    print("[3/3] 开始训练...")
    print("-"*80)
    
    total_timesteps = 500000
    print(f"总训练步数: {total_timesteps:,}")
    print(f"预计时间: 10-30分钟 (取决于CPU/GPU)\n")
    
    model.learn(total_timesteps=total_timesteps)
    
    print("\n" + "-"*80)
    model.save(model_path)
    print(f"✓ 训练完成！模型已保存: {model_path}.zip")
    print("="*80 + "\n")
