import random
import gymnasium as gym
import numpy as np
import ale_py
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from DQN_agent import DQN, ReplayBuffer

def continue_training(model_path, additional_episodes=400):
    # 超参数
    lr = 2e-3
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01  # 可以设置较小的探索率，因为模型已经有一定的经验
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    save_interval = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    # env = gym.make("ALE/Pong-v5", render_mode="human")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    obs, info = env.reset()
    
    # 设置随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 创建智能体和经验回放池
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 加载已有模型
    checkpoint = torch.load(model_path, map_location=device)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.target_q_net.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    return_list = checkpoint.get('return_list', [])
    
    print(f"Continuing training from episode {start_episode}")
    
    # 继续训练
    total_reward = 0
    for i in range(10):
        with tqdm(total=int(additional_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(additional_episodes / 10)):
                episode_return = 0
                state, info = env.reset(seed=0)
                done = False
                
                while not done:
                    # env.render()
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    
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
                current_episode = start_episode + (additional_episodes/10 * i + i_episode + 1)
                
                # 定期保存模型
                if (i_episode + 1) % save_interval == 0:
                    save_path = f'models/dqn_pong_episode_{current_episode}.pth'
                    torch.save({
                        'episode': current_episode,
                        'model_state_dict': agent.q_net.state_dict(),
                        'target_model_state_dict': agent.target_q_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'return_list': return_list,
                    }, save_path)
                    print(f'\nModel saved to {save_path}')
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % current_episode,
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    print(f"Total reward in continued training: {total_reward}")
    
    # 保存最终模型
    final_save_path = f'models/dqn_pong_episode_{start_episode + additional_episodes}.pth'
    torch.save({
        'episode': start_episode + additional_episodes,
        'model_state_dict': agent.q_net.state_dict(),
        'target_model_state_dict': agent.target_q_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'return_list': return_list,
    }, final_save_path)
    print(f'\nFinal model saved to {final_save_path}')

if __name__ == "__main__":
    # 使用之前保存的模型继续训练
    model_path = "models/dqn_pong_final.pth"  # 或者使用其他检查点
    continue_training(model_path, additional_episodes=400)  # 继续训练100个episodes 