 import gymnasium as gym
 import numpy as np
 import ale_py
 gym.register_envs(ale_py)
 env = gym.make("ALE/Pong-v5", render_mode="human")
 obs = env.reset()
 total_reward  0
 for _ in range(1000):
 env.render()
 action = env.action_space.sample() 
obs, reward, done, truncated, info = env.step(action)
 total_reward += reward
 print(f"Action: {action}, Reward: {reward}, Done: {done}")
 if done:
 obs = env.reset()