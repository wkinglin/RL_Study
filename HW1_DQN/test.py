import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ale_py

def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float32).ravel()  # 确保使用 np.float32 以避免警告

def show_image(status):
    """ 处理和显示 Pong 游戏的帧图像 """

    status1 = status[35:195]  # 裁剪有效区域
    status2 = status1[::2, ::2, 0]  # 下采样

    # 统计像素点构成
    dict_color = Counter(status2.ravel())
    print("像素点构成: ", dict_color)

    # 擦除背景并转换为灰度
    status3 = status2.copy()
    status3[status3 == 144] = 0
    status3[status3 == 109] = 0
    status3[status3 != 0] = 1  # 转换为二值图像

    # 可视化操作中间图
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    titles = ["Original", "Cropped", "Downsampled", "Processed"]
    images = [status, status1, status2, status3]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=plt.cm.binary)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# 创建 Pong 训练环境
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# 获取初始状态
status, _ = env.reset()  # 需要解包，因为 Gymnasium 版本返回 (observation, info)
show_image(status)
