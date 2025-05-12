import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from MBPO import PolicyNet, QValueNet, SAC, EnsembleDynamicsModel, FakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_dir="MBPO_models_final"):
    """加载保存的模型参数"""
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"找不到模型目录: {model_dir}")
    
    # 加载超参数
    with open(f"{model_dir}/hyperparams.json", 'r') as f:
        hyperparams = json.load(f)
    
    print("加载超参数:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # 创建环境
    env_name = hyperparams["env_name"]
    env = gym.make(env_name, render_mode="human", terminate_when_unhealthy=False)
    env.reset(seed=0)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
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
    print("\n加载模型参数...")
    agent.actor.load_state_dict(torch.load(f"{model_dir}/sac_actor.pth", map_location=device))
    print("策略网络加载成功")
    agent.critic_1.load_state_dict(torch.load(f"{model_dir}/sac_critic_1.pth", map_location=device))
    agent.critic_2.load_state_dict(torch.load(f"{model_dir}/sac_critic_2.pth", map_location=device))
    print("值网络加载成功")
    agent.target_critic_1.load_state_dict(torch.load(f"{model_dir}/sac_target_critic_1.pth", map_location=device))
    agent.target_critic_2.load_state_dict(torch.load(f"{model_dir}/sac_target_critic_2.pth", map_location=device))
    agent.log_alpha = torch.load(f"{model_dir}/sac_log_alpha.pth", map_location=device)
    print("SAC参数加载成功")
    
    try:
        model.model.load_state_dict(torch.load(f"{model_dir}/dynamics_model.pth", map_location=device))
        print("环境模型加载成功")
    except Exception as e:
        print(f"环境模型加载失败: {e}")
    
    return env, agent, model, hyperparams

def evaluate(env, agent, num_episodes=10, render=False):
    """评估智能体性能"""
    returns = []
    steps_list = []
    
    print(f"\n开始评估，共{num_episodes}个episode...")
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
        steps_list.append(steps)
        print(f"Episode {i+1}: 回报 = {episode_return:.2f}, 步数 = {steps}")
    
    avg_return = np.mean(returns)
    avg_steps = np.mean(steps_list)
    std_return = np.std(returns)
    print(f"\n评估结果:")
    print(f"平均回报: {avg_return:.2f} ± {std_return:.2f}")
    print(f"平均步数: {avg_steps:.2f}")
    return returns, steps_list

if __name__ == "__main__":
    env = None
    model_dir = "MBPO_models"
    try:
        # 加载模型
        env, agent, model, hyperparams = load_model(model_dir=model_dir)
        print(1)
        # 评估模型
        returns, steps = evaluate(env, agent, num_episodes=10, render=True)
        print(2)
        # 可视化评估结果
        plt.figure(figsize=(12, 5))
        
        # 绘制回报
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(returns) + 1), returns)
        plt.axhline(y=np.mean(returns), color='r', linestyle='-', 
                   label=f'平均回报: {np.mean(returns):.2f}±{np.std(returns):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'回报评估 ({hyperparams["env_name"]})')
        plt.legend()
        
        # 绘制步数
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(steps) + 1), steps)
        plt.axhline(y=np.mean(steps), color='r', linestyle='-', 
                   label=f'平均步数: {np.mean(steps):.2f}±{np.std(steps):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('步数评估')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{model_dir}/evaluation_results.png", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
    finally:
        # 确保环境被正确关闭
        if env is not None:
            env.close() 