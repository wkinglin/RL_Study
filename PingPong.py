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
from DQN_agent import DQN, ReplayBuffer
import logging

def saveModel(num_episodes,agent,return_list,final_save_path):
    # 保存最终的模型
    torch.save({
        'episode': num_episodes,
        'model_state_dict': agent.q_net.state_dict(),
        'target_model_state_dict': agent.target_q_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'return_list': return_list,
    }, final_save_path)
    print(f'\n model saved to {final_save_path}')

if __name__ == "__main__":
    # 超参数
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start

    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    total_reward = 0
    save_interval = 50  # 每100个episode保存一次模型

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gym.register_envs(ale_py)
    # env = gym.make("ALE/Pong-v5", render_mode="human")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    obs, info = env.reset() #obs是numpy数组

    # 设置日志记录，filemode='w'表示每次启动训练时覆盖之前的日志
    logging.basicConfig(filename='training_log.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

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
                # 更新 epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                agent.epsilon = epsilon
                logging.info(f'Episode {i_episode + 1}, Epsilon: {epsilon}')

                episode_return = 0
                state, info = env.reset(seed=0)
                done = False

                while not done:
                    # 渲染环境
                    # env.render()

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
                        loss = agent.update(transition_dict)  # 假设 update 方法返回损失值
                        logging.info(f'Episode {i_episode + 1}, Loss: {loss}')      
                # print(f'Episode {i_episode + 1}, Loss: {loss}')
                return_list.append(episode_return)
                
                # 定期保存模型
                if (i_episode + 1) % save_interval == 0:
                    save_path = f'models/dqn_pong_episode_{num_episodes/10 * i + i_episode + 1}.pth'
                    saveModel(num_episodes/10 * i + i_episode + 1,agent,return_list,save_path)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                    logging.info(f'Iteration {i}, Average Return: {np.mean(return_list[-10:])}')
                pbar.update(1)
        
    print(f"Total reward: {total_reward}")
    final_save_path = 'models/dqn_pong_final.pth'
    saveModel(num_episodes,agent,return_list,final_save_path)

    # 训练结束后关闭日志记录
    logging.shutdown()
