import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from MBPO import PolicyNet, QValueNet, SAC, EnsembleDynamicsModel, FakeEnv

def load_model(model_dir="saved_models"):
    """加载保存的模型参数"""
    # 加载超参数
    with open(f"{model_dir}/hyperparams.json", 'r') as f:
        hyperparams = json.load(f)
    
    print("加载超参数:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # 创建环境
    env_name = hyperparams["env_name"]
    env = gym.make(env_name)
    env.reset(seed=0)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建智能体和模型
    agent = SAC(
        state_dim=state_dim,
        hidden_dim=hyperparams["hidden_dim"],
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=hyperparams["actor_lr"],
        critic_lr=hyperparams["critic_lr"],
        alpha_lr=hyperparams["alpha_lr"],
        target_entropy=hyperparams["target_entropy"],
        tau=hyperparams["tau"],
        gamma=hyperparams["gamma"]
    )
    
    model = EnsembleDynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        model_alpha=hyperparams["model_alpha"]
    )
    
    # 加载模型参数
    agent.actor.load_state_dict(torch.load(f"{model_dir}/sac_actor.pth", map_location=device))
    agent.critic_1.load_state_dict(torch.load(f"{model_dir}/sac_critic_1.pth", map_location=device))
    agent.critic_2.load_state_dict(torch.load(f"{model_dir}/sac_critic_2.pth", map_location=device))
    agent.target_critic_1.load_state_dict(torch.load(f"{model_dir}/sac_target_critic_1.pth", map_location=device))
    agent.target_critic_2.load_state_dict(torch.load(f"{model_dir}/sac_target_critic_2.pth", map_location=device))
    agent.log_alpha = torch.load(f"{model_dir}/sac_log_alpha.pth", map_location=device)
    
    try:
        model.model.load_state_dict(torch.load(f"{model_dir}/dynamics_model.pth", map_location=device))
        print("环境模型加载成功")
    except Exception as e:
        print(f"环境模型加载失败: {e}")
    
    return env, agent, model, hyperparams

def evaluate(env, agent, num_episodes=10, render=False):
    """评估智能体性能"""
    returns = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        steps = 0
        
        while not done:
            action = agent.take_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            obs = next_obs
            steps += 1
            
            if render:
                env.render()
        
        returns.append(episode_return)
        print(f"Episode {i+1}: Return = {episode_return}, Steps = {steps}")
    
    avg_return = np.mean(returns)
    print(f"平均回报: {avg_return:.2f}")
    return returns

if __name__ == "__main__":
    # 加载模型
    env, agent, model, hyperparams = load_model()
    
    # 评估模型
    print("\n开始评估...")
    returns = evaluate(env, agent, num_episodes=5)
    
    # 可视化评估结果
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(returns) + 1), returns)
    plt.axhline(y=np.mean(returns), color='r', linestyle='-', label=f'平均回报: {np.mean(returns):.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Model Evaluation on {hyperparams["env_name"]}')
    plt.legend()
    plt.savefig("evaluation_results.png")
    plt.show() 