import gymnasium as gym
import torch
import numpy as np
import ale_py
from DQN_agent import DQN  # 导入DQN类

def play_pong(model_path, num_episodes=5):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    env = gym.make("ALE/Pong-v5", render_mode="human")  # 使用human模式来显示游戏画面
    
    # 获取动作空间
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = DQN(
        state_dim=env.observation_space.shape,
        hidden_dim=128,
        action_dim=action_dim,
        learning_rate=2e-3,
        gamma=0.98,
        epsilon=0.01,  # 在测试时使用较小的探索率
        target_update=10,
        device=device
    )
    
    # 加载训练好的模型
    checkpoint = torch.load(model_path, map_location=device)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.target_q_net.load_state_dict(checkpoint['target_model_state_dict'])
    print(f"Loaded model from episode {checkpoint['episode']}")
    
    # 设置为评估模式
    agent.q_net.eval()
    agent.target_q_net.eval()
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            env.render()

            # 选择动作
            action = agent.take_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # 使用保存的最终模型
    model_path = "models/dqn_pong_final_500.pth"
    play_pong(model_path) 