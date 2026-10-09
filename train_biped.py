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


class TrainingMonitor:
    def __init__(self):
        self.rewards = []
        self.episode_lengths = []
        self.times = []
        self.start_time = time.time()
        self.start_datetime = datetime.now()

    def update(self, reward, episode_length):
        elapsed = time.time() - self.start_time
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.times.append(elapsed)

    def plot(self):
        if len(self.rewards) < 2:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"双足机器人舞蹈训练 - {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=15, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(self.times, self.rewards, 'b-', linewidth=2, label='episode reward')
        if len(self.rewards) > 5:
            rolling = np.convolve(self.rewards, np.ones(5)/5, mode='valid')
            ax1.plot(self.times[4:], rolling, 'r--', linewidth=2, label='5-step moving avg')
        ax1.set_xlabel('Elapsed Time (s)')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Curve')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        ax2 = axes[0, 1]
        ax2.plot(self.times, self.episode_lengths, 'g-', linewidth=2)
        ax2.set_xlabel('Elapsed Time (s)')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length')
        ax2.grid(alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.hist(self.rewards, bins=25, color='skyblue', edgecolor='black', alpha=0.8)
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Count')
        ax3.set_title('Reward Distribution')
        ax3.grid(alpha=0.2, axis='y')
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats = f"""
训练统计:
- Episodes: {len(self.rewards)}
- Avg Reward: {np.mean(self.rewards):.2f}
- Max Reward: {np.max(self.rewards):.2f}
- Min Reward: {np.min(self.rewards):.2f}
- Avg Length: {np.mean(self.episode_lengths):.1f}
- Max Length: {np.max(self.episode_lengths):.1f}
- Elapsed: {(time.time() - self.start_time)/60:.1f} min
        """
        ax4.text(0.05, 0.5, stats, fontsize=12, family='monospace',
                 bbox=dict(facecolor='wheat', alpha=0.5, pad=12))
        
        plt.tight_layout()
        plt.pause(0.001)
        return fig


def train_with_popup():
    print("="*80)
    print("双足机器人舞蹈训练（PPO + PyBullet + 实时弹窗）")
    print("="*80)
    
    monitor = TrainingMonitor()
    
    print("[1/3] 创建环境...")
    env = DummyVecEnv([lambda: UnitreeBipedDanceEnv(render_mode=None)])
    
    model_path = "unitree_biped_dance_ppo"
    
    if os.path.exists(model_path + ".zip"):
        print("发现已有模型，继续训练...")
        model = PPO.load(model_path)
    else:
        print("未发现模型，从零开始训练...")
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
    
    print("[2/3] 准备训练窗口...")
    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    fig.show()
    
    print("[3/3] 开始训练...")
    print("-" * 80)
    
    total_rounds = 20
    steps_per_round = 50000
    
    for round_idx in range(1, total_rounds + 1):
        print(f"\n第 {round_idx}/{total_rounds} 轮训练开始...")
        model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        
        obs, _ = env.envs[0].reset()
        episode_reward = 0.0
        episode_length = 0
        
        for _ in range(1500):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.envs[0].step(action)
            episode_reward += reward
            episode_length += 1
            if done or truncated:
                break
        
        monitor.update(episode_reward / episode_length if episode_length > 0 else 0, episode_length)
        model.save(model_path)
        
        print(f"✓ 第 {round_idx} 轮训练完成 | reward={monitor.rewards[-1]:.2f} | steps={episode_length}")
        
        fig.clear()
        monitor.plot()
        plt.pause(0.1)
    
    print("\n" + "="*80)
    print("✓ 训练完成，模型已保存：", model_path + ".zip")
    print("="*80)
    
    fig.clear()
    monitor.plot()
    plt.show()


if __name__ == "__main__":
    train_with_popup()
