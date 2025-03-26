import gymnasium as gym
import ale_py
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os

# 创建模型保存目录
models_dir = "SB3_DDQN_models/" 

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # 从经验回放缓冲区采样数据
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # 不对目标网络进行梯度反向传播
            with torch.no_grad():
                # 使用目标网络计算下一个状态的Q值
                next_q_values = self.q_net_target(replay_data.next_observations)
                # 使用在线网络计算下一个状态的Q值（用于动作选择）
                next_q_values_online = self.q_net(replay_data.next_observations)
                # 使用在线网络选择动作
                next_actions_online = next_q_values_online.argmax(dim=1, keepdim=True)
                # 使用目标网络估计所选动作的Q值
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions_online)
               
                # 计算1步TD目标
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # 获取当前状态的Q值估计
            current_q_values = self.q_net(replay_data.observations)

            # 获取回放缓冲区中动作对应的Q值
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # 检查形状是否匹配
            assert current_q_values.shape == target_q_values.shape

            # 计算损失（使用Huber损失）
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            losses.append(loss.item())

            # 优化Q网络
            self.policy.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # 增加更新计数器
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# 训练 Double DQN 模型
model = DoubleDQN(
    "CnnPolicy", 
    env, 
    verbose=1, 
    buffer_size=10000,  # 增加经验回放缓冲区
    learning_rate=1e-4,  # 降低学习率，稳定训练
    exploration_fraction=0.1,  # 控制探索衰减速率
    exploration_final_eps=0.01,  # 最终探索率
    target_update_interval=1000,  # 目标网络更新频率
    train_freq=4,  # 训练频率
    gradient_steps=1,  # 每次训练的梯度更新步数
    batch_size=128,  # 训练时使用的批次大小
    tensorboard_log="Sb3_DDQN_pong_log"
)

# 评估代码
ean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20,
)

# 训练 500000 个时间步
for i in range(5):
    model.learn(total_timesteps=100000, log_interval=4, reset_num_timesteps=False)
    model.save(f"{models_dir}/ddqn_pong_step_{i}")

# 保存最终模型
model.save(f"{models_dir}/ddqn_pong_final")

# del model # remove to demonstrate saving and loading

# model = DDQN.load("stableBaseline3Model/dqn_pong")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
