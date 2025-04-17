import json
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
with open('./PPO_total_reward.json', 'r') as f:
    data = json.load(f)

    # 读取数据
with open('./PPO_episode_count.json', 'r') as f:
    data_episode = json.load(f)

# 提取时间戳、步数和奖励
time = [item[0] for item in data]
steps = [item[1] for item in data]
rewards = [item[2] for item in data]

episode_count = [item[2] for item in data_episode]


# 删除episode_count和rewards中的重复元素
unique_episodes = []
unique_rewards = []
seen_episodes = set()

for i, episode in enumerate(episode_count):
    if episode not in seen_episodes:
        seen_episodes.add(episode)
        unique_episodes.append(episode)
        unique_rewards.append(rewards[i])

# 更新数据
episode_count = unique_episodes
rewards = unique_rewards

# 确保数据长度一致
min_length = min(len(episode_count), len(rewards))
episode_count = episode_count[:min_length]
rewards = rewards[:min_length]

# 绘制奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(episode_count, rewards, 'b-', label='Reward')

# 添加移动平均线
window_size = 10
rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
episode_count_ma = episode_count[window_size-1:]
# plt.plot(episode_count_ma, rewards_ma, 'r-', label=f'{window_size}-Episode Moving Average')

# 设置图表
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards over Episodes')
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('PPO_reward_episode.png')
plt.show()

# 打印一些统计信息
print(f"Total episodes: {len(episode_count)}")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Max reward: {np.max(rewards):.2f}")
print(f"Min reward: {np.min(rewards):.2f}")
