import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import ale_py
import os
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, ResizeObservation, TransformObservation

class CropAndNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, crop_box):
        super().__init__(env)
        self.crop_box = crop_box
        # 更新 observation_space 以反映裁剪后的图像尺寸
        old_shape = self.observation_space.shape
        if(len(old_shape) == 3):
            new_shape = (old_shape[2], self.crop_box[2] - self.crop_box[0], self.crop_box[3] - self.crop_box[1])
        else:
            new_shape = (self.crop_box[2] - self.crop_box[0], self.crop_box[3] - self.crop_box[1])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        # 对图像进行裁剪和标准化
        cropped = obs[self.crop_box[0]:self.crop_box[2], self.crop_box[1]:self.crop_box[3]]
        return cropped.astype(np.float32) / 255.0  # 标准化

def process_image(env):
    env = GrayscaleObservation(env, keep_dim=True)  # 转换为灰度图像
    env = ResizeObservation(env, (110, 84))  # 210x160缩放到84x110
    env = CropAndNormalizeObservation(env, crop_box=(18, 0, 102, 84))  # 裁剪区域：(y_start, x_start, y_end, x_end)
    # env = FrameStackObservation(env, 4)  # 堆叠4帧图像 4x110x84
    print(env.observation_space)
    env = TransformObservation(env, lambda x: np.squeeze(x).transpose(2, 0, 1) if len(x.shape) == 3 else x, env.observation_space)
    env.reset()
    print(env.observation_space)
    # env = TransformObservation(env, lambda x: x.transpose(2, 0, 1), env.observation_space)
    return env

# 创建模型保存目录
models_dir = "SB3_DQN_models/pictureProcess1"
log_dir = "Sb3_DQN_pong_log/pictureProcess1"

model_name = "dqn_pong_pictureProcess1"
continue_model_pos = "0"
one_timestep = 500000
total_turn = 2

# 创建环境并应用包装器
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")  # 训练时使用rgb_array模式
env = process_image(env)

# 检查观察空间是否符合要求
obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Observation shape: {obs.shape}")


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
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, tensorboard_log=log_dir,policy_kwargs={"normalize_images": False})
    # 训练模型并保存中间结果
    for i in range(total_turn):
        model.learn(total_timesteps=one_timestep, log_interval=4, reset_num_timesteps=False)
        model.save(f"{models_dir}/{model_name}_step_{i}")

def continue_train_model():
    model = DQN.load(f"{models_dir}/{model_name}_step_{continue_model_pos}.zip", env=env)
    for i in range(total_turn):
        model.learn(total_timesteps=one_timestep, log_interval=4, reset_num_timesteps=False)
        model.save(f"{models_dir}/{model_name}_step_{continue_model_pos+i}")
    model.save(f"{models_dir}/{model_name}_final")


# continue_train_model()
# test_model(f"{models_dir}/dqn_pong_final")
# train_model()





