import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
np.random.seed(0)
env = gym.make("CartPole-v1")
env.reset(seed=42)
print(env.observation_space)
env = gym.make("CartPole-v1")
env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
env.reset(seed=42)
print(env.observation_space)
