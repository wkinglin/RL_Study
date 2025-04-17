import json
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
with open('./DDQN_loss.json', 'r') as f:
    data = json.load(f)

# 提取时间戳、步数和奖励
timestamps = [item[0] for item in data]
steps = [item[1] for item in data]
rewards = [item[2] for item in data]

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制奖励曲线
plt.plot(steps, rewards, 'b-', label='Loss')

# 设置标题和标签
plt.title('Training Loss Over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')

# 添加网格
plt.grid(True)

# 添加图例
plt.legend()

# 保存图片
plt.savefig('DDQN_loss.png')
plt.close()

# 打印一些统计信息
print(f"Total steps: {steps[-1]}")
print(f"Final reward: {rewards[-1]}")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Max reward: {max(rewards):.2f}")
print(f"Min reward: {min(rewards):.2f}")