import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import argparse

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_rewards = []
        self.episode_count = 0
        self.total_reward = 0
        self.reward_return = 0

    def _on_step(self):
        # 获取当前step的reward
        reward = self.locals['rewards'][0]
        self.total_reward += reward
        
        # 检查是否episode结束
        done = self.locals['dones'][0]
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.total_reward)
            self.reward_return += self.total_reward
            self.total_reward = 0
            
            # 记录到tensorboard
            self.logger.record("train/total_reward", self.reward_return)
            self.logger.record("train/episode_reward", self.episode_rewards[-1])
            self.logger.record("train/episode_count", self.episode_count)
            self.logger.record("train/mean_reward", np.mean(self.episode_rewards))
            self.logger.record("train/std_reward", np.std(self.episode_rewards))
            
        return True


def train_ppo(model_path, one_timestep, total_turn):
    # 创建回调
    reward_callback = RewardCallback()
    
    # 开始训练
    for i in range(total_turn):
        model.learn(
            total_timesteps=one_timestep, 
            log_interval=4, 
            reset_num_timesteps=False,
            callback=reward_callback
        )
        model.save(f"{model_path}_step_{i}")
        
        # 评估当前模型
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        print(f"==> After step {i+1}, mean_reward: {mean_reward}, std: {std_reward}")


def continue_train(model_path,one_timestep,total_turn,continue_model_pos):
    reward_callback = RewardCallback()
    model = PPO.load(f"{model_path}_step_{continue_model_pos}.zip", env=env)
    for i in range(total_turn):
        model.learn(total_timesteps=one_timestep, log_interval=4, reset_num_timesteps=False, callback=reward_callback)
        model.save(f"{model_path}_step_{continue_model_pos + i + 1}")

def test_ppo(model_path,step):
    # 重新加载模型
    model = PPO.load(f"{model_path}_step_{step}")

    # 测试模型表现
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        time.sleep(0.05)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Training Parameters')
    
    # 训练参数
    parser.add_argument('--train', type=bool, default=True,
                      help='是否训练')
    parser.add_argument('--continue_train', type=bool, default=False,
                      help='是否继续训练')
    parser.add_argument('--continue_model_pos', type=int, default=1,
                      help='继续训练的模型位置')
    parser.add_argument('--test', type=bool, default=False,
                      help='是否测试')
    parser.add_argument('--one_timesteps', type=int, default=500000,
                      help='每轮的训练步数')
    parser.add_argument('--total_turn', type=int, default=4,
                      help='总训练轮数')
    
    # 验证参数
    if parser.parse_args().continue_train and parser.parse_args().continue_model_pos is None:
        parser.error("当continue_train为True时，必须指定continue_model_pos")
        
    # 其他参数
    parser.add_argument('--tensorboard_log', type=str, default='./PPO_tensorboard/ppo_halfcheetah_tensorboard_change',
                      help='tensorboard日志目录')
    parser.add_argument('--save_path', type=str, default='./PPO_model/ppo_halfcheetah_model_change',
                      help='模型保存路径')

    total_turn = parser.parse_args().total_turn
    one_timestep = parser.parse_args().one_timesteps

    tensorboard_path = parser.parse_args().tensorboard_log
    model_path = parser.parse_args().save_path
    continue_model_pos = parser.parse_args().continue_model_pos

    # 创建并包装环境
    env = gym.make("HalfCheetah-v5", exclude_current_positions_from_observation = False)
    env = DummyVecEnv([lambda: env])

    # 初始化 PPO 模型，使用 MLP 策略
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path)

    if parser.parse_args().train:
        train_ppo(model_path,one_timestep,total_turn)
    elif parser.parse_args().continue_train:
        continue_train(model_path,one_timestep,total_turn,continue_model_pos)
    elif parser.parse_args().test:
        test_ppo(model_path,continue_model_pos)