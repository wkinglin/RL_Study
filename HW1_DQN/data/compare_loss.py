import json
import matplotlib.pyplot as plt
import numpy as np

# 读取DQN数据
with open('./DQN_loss.json', 'r') as f:
    dqn_data = json.load(f)

# 读取DDQN数据
with open('./DDQN_loss.json', 'r') as f:
    ddqn_data = json.load(f)

# 提取数据
dqn_steps = [item[1] for item in dqn_data]
dqn_rewards = [item[2] for item in dqn_data]

ddqn_steps = [item[1] for item in ddqn_data]
ddqn_rewards = [item[2] for item in ddqn_data]

# 创建图形
plt.figure(figsize=(15, 8))

# 绘制两条曲线
plt.plot(dqn_steps, dqn_rewards, 'b-', label='DQN', linewidth=2)
plt.plot(ddqn_steps, ddqn_rewards, 'r-', label='DDQN', linewidth=2)

# 设置标题和标签
plt.title('DQN vs DDQN Training Loss Comparison', fontsize=14)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 设置x轴范围
# plt.xlim(0, 2e6)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=12)

# 设置坐标轴字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 保存图片
plt.savefig('dqn_ddqn_comparison_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印统计信息
print("DQN Statistics:")
print(f"Total steps: {dqn_steps[-1]}")
print(f"Final reward: {dqn_rewards[-1]:.2f}")
print(f"Average reward: {np.mean(dqn_rewards):.2f}")
print(f"Max reward: {max(dqn_rewards):.2f}")
print(f"Min reward: {min(dqn_rewards):.2f}")
print("\nDDQN Statistics:")
print(f"Total steps: {ddqn_steps[-1]}")
print(f"Final reward: {ddqn_rewards[-1]:.2f}")
print(f"Average reward: {np.mean(ddqn_rewards):.2f}")
print(f"Max reward: {max(ddqn_rewards):.2f}")
print(f"Min reward: {min(ddqn_rewards):.2f}")
