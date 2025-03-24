import random
import gymnasium as gym
import numpy as np
import ale_py
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from PIL import Image  # 确保在文件顶部导入
import os

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 计算卷积后的特征图大小
        # 输入: 210x160
        # conv1: (210-8)/4 + 1 = 51, (160-8)/4 + 1 = 39
        # conv2: (51-4)/2 + 1 = 24, (39-4)/2 + 1 = 18
        # conv3: (24-3)/1 + 1 = 22, (18-3)/1 + 1 = 16
        self.fc4 = torch.nn.Linear(64 * 22 * 16, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc4(x))
        return self.head(x)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # Q网络
        # self.q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)  
        self.q_net = ConvolutionalQnet(self.action_dim, 3).to(device)  

        # 目标网络
        # self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)
        self.target_q_net = ConvolutionalQnet(self.action_dim, 3).to(device)  
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # 确保state是numpy数组
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            # 打印state的形状以进行调试
            # print("Original state shape:", state.shape)
            
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            # print("State tensor shape before permute:", state_tensor.shape)
            
            # 调整维度顺序从(H, W, C)到(C, H, W)并添加batch维度
            state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
            # print("State tensor shape after permute:", state_tensor.shape)
            
            action = self.q_net(state_tensor).argmax().item()
        return action

    def update(self, transition_dict):
        # 处理状态维度
        states = torch.tensor(transition_dict['states'],
                            dtype=torch.float).to(self.device)
        # 调整维度从(batch, H, W, C)到(batch, C, H, W)
        states = states.permute(0, 3, 1, 2)
        
        next_states = torch.tensor(transition_dict['next_states'],
                                 dtype=torch.float).to(self.device)
        # 同样调整next_states的维度
        next_states = next_states.permute(0, 3, 1, 2)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                           dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

# 超参数
lr = 2e-3
num_episodes = 100
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
total_reward = 0
save_interval = 100  # 每100个episode保存一次模型

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

gym.register_envs(ale_py)
# env = gym.make("ALE/Pong-v5", render_mode="human")
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
obs, info = env.reset() #obs是numpy数组

random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)

# 创建保存模型的目录
if not os.path.exists('models'):
    os.makedirs('models')

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape
# state_dim = 84 * 84
action_dim = env.action_space.n
print(state_dim)
print(action_dim)

agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # env.render()
            episode_return = 0
            state, info = env.reset(seed=0)
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            
            # 定期保存模型
            if (i_episode + 1) % save_interval == 0:
                save_path = f'models/dqn_pong_episode_{num_episodes/10 * i + i_episode + 1}.pth'
                torch.save({
                    'episode': num_episodes/10 * i + i_episode + 1,
                    'model_state_dict': agent.q_net.state_dict(),
                    'target_model_state_dict': agent.target_q_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'return_list': return_list,
                }, save_path)
                print(f'\nModel saved to {save_path}')
            
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    
print(f"Total reward: {total_reward}")

# 保存最终的模型
final_save_path = 'models/dqn_pong_final.pth'
torch.save({
    'episode': num_episodes,
    'model_state_dict': agent.q_net.state_dict(),
    'target_model_state_dict': agent.target_q_net.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'return_list': return_list,
}, final_save_path)
print(f'\nFinal model saved to {final_save_path}')
