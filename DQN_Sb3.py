import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import os

# 创建模型保存目录
models_dir = "SB3_DQN_models/"
log_dir = "Sb3_DQN_pong_log/"
# env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = gym.make("ALE/Pong-v5", render_mode="human")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 推理模型
def test_model(path):
    model = DQN.load(path)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def train_model():
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, tensorboard_log=log_dir)
    # 训练模型并保存中间结果
    for i in range(10):
        model.learn(total_timesteps=100000, log_interval=4, reset_num_timesteps=False)
        model.save(f"{models_dir}/dqn_pong_step_{i}")

def continue_train_model():
    model = DQN.load(f"{models_dir}/dqn_pong_step_20.zip", env=env)
    for i in range(10):
        model.learn(total_timesteps=100000, log_interval=4, reset_num_timesteps=False)
        model.save(f"{models_dir}/dqn_pong_step_{10+i}")
    model.save(f"{models_dir}/dqn_pong_final")

# continue_train_model()
test_model(f"{models_dir}/dqn_pong_final")




