import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time


def train_ppo(model_path,one_timestep,total_turn):
    # 开始训练
    for i in range(total_turn):
        model.learn(total_timesteps=one_timestep, log_interval=4, reset_num_timesteps=False)
        model.save(f"{model_path}_step_{i}")

        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        # print(f"==> After step {i+1}, mean_reward: {mean_reward}, std: {std_reward}")

def continue_train(model_path,one_timestep,total_turn,continue_model_pos):
    model = PPO.load(f"{model_path}_step_{continue_model_pos}.zip", env=env)
    for i in range(total_turn):
        model.learn(total_timesteps=one_timestep, log_interval=4, reset_num_timesteps=False)
        model.save(f"{model_path}_step_{i}")

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

total_turn = 4
one_timestep = 500000
model_path = "./PPO_model/ppo_halfcheetah_model_change"
tensorboard_path = "./PPO_tensorboard/ppo_halfcheetah_tensorboard_change"

# 创建并包装环境
env = gym.make("HalfCheetah-v5", exclude_current_positions_from_observation = False)
env = DummyVecEnv([lambda: env])

# 初始化 PPO 模型，使用 MLP 策略
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path)

train_ppo(model_path,one_timestep,total_turn)
# continue_train(model_path,one_timestep,total_turn,continue_model_pos=3)
# test_ppo(model_path,3)