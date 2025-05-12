import numpy as np
import torch
import torch.serialization
from torch.serialization import add_safe_globals
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import json
from MPC_PETS import PETS, ReplayBuffer, CEM, FakeEnv, EnsembleDynamicsModel

# 添加NumPy的scalar类型到安全列表
from numpy.core.multiarray import scalar
add_safe_globals([scalar])

def create_eval_dir():
    """创建评估结果目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_dir = f"evaluations/compare_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir

def safe_load_model(pets_instance, model_path):
    """安全加载模型的自定义函数，处理PyTorch 2.6+的兼容性问题"""
    try:
        print(f"尝试加载模型: {model_path}")
        # 先尝试使用weights_only=True加载
        save_dict = torch.load(model_path)
        print("成功使用weights_only=True加载模型")
    except Exception as e:
        print(f"使用weights_only=True加载失败，尝试使用weights_only=False: {e}")
        # 如果失败，使用weights_only=False再次尝试
        try:
            save_dict = torch.load(model_path, weights_only=False)
            print("成功使用weights_only=False加载模型")
        except Exception as e2:
            print(f"所有加载方式都失败: {e2}")
            raise e2
    
    # 加载模型参数
    if 'model_state' in save_dict:
        pets_instance._model.model.load_state_dict(save_dict['model_state'])
        print("成功加载模型状态字典")
    else:
        print("警告: 加载的文件中没有找到'model_state'键")
    
    # 返回超参数（如果有）
    if 'hyperparams' in save_dict:
        print("已加载超参数")
        return save_dict['hyperparams']
    else:
        print("未找到超参数，使用默认值")
        return None

def evaluate_pets_model(model_path, env_name="Hopper-v5", num_episodes=5, render=False):
    """纯评估模式加载预训练PETS模型"""
    print(f"开始评估PETS模型: {model_path}")
    
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode="human", terminate_when_unhealthy=False )
    else:
        env = gym.make(env_name)
    
    # 创建空的回放缓冲区 - 仅用于初始化PETS，不会实际用于收集数据
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # 创建PETS实例（参数随后会被加载的模型覆盖）
    pets = PETS(
        env=env,
        replay_buffer=replay_buffer,
        n_sequence=50,
        elite_ratio=0.2,
        plan_horizon=25,
        num_episodes=0  # 不需要训练，只用于评估
    )
    
    # 使用安全的方式加载模型和超参数
    hyperparams = safe_load_model(pets, model_path)
    print("模型加载完成，开始评估...")
    
    # 评估模型
    returns = []
    
    for episode in range(num_episodes):
        print(f"PETS评估 Episode {episode+1}/{num_episodes}")
        
        # 手动执行评估，而不是调用pets.mpc()
        obs, _ = env.reset(seed=episode + 100)  # 使用不同的种子确保多样性
        episode_return = 0
        done = False
        step = 0
        
        # 初始化均值和方差
        plan_horizon = pets.plan_horizon
        mean = np.zeros(plan_horizon * 3)
        for i in range(plan_horizon):
            mean[i*3:(i+1)*3] = (pets.upper_bound + pets.lower_bound) / 2.0
        var = np.zeros(plan_horizon * 3)
        for i in range(plan_horizon):
            var[i*3:(i+1)*3] = np.square(pets.upper_bound - pets.lower_bound) / 16
        
        while not done:
            step += 1
            
            # 仅使用CEM进行规划，不添加到回放缓冲区
            actions = pets._cem.optimize(obs, mean, var)
            action = actions[0]  # 选取第一个动作
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 累计奖励
            episode_return += reward
            
            # 更新状态
            obs = next_obs
            
            # 更新mean，移除已执行的动作并在末尾添加零动作
            mean = np.zeros(plan_horizon * 3)
            for i in range(plan_horizon - 1):
                mean[i*3:(i+1)*3] = actions[i+1] if i+1 < len(actions) else ((pets.upper_bound + pets.lower_bound) / 2.0)
            
            # 同样更新方差
            var = np.zeros(plan_horizon * 3)
            for i in range(plan_horizon):
                var[i*3:(i+1)*3] = np.square(pets.upper_bound - pets.lower_bound) / 16
            
            # 定期打印状态
            if step % 20 == 0:
                print(f"  步骤 {step}，当前奖励: {reward:.2f}，累计奖励: {episode_return:.2f}")
        
        print(f"Episode {episode+1} 完成，总步数: {step}，总回报: {episode_return:.2f}")
        returns.append(episode_return)
    
    # 计算结果统计
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"\nPETS模型评估完成!")
    print(f"平均回报: {mean_return:.2f} ± {std_return:.2f}")
    
    return {
        "returns": returns,
        "mean_return": mean_return,
        "std_return": std_return,
        "num_episodes": num_episodes
    }

def compare_models():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建评估目录
    eval_dir = create_eval_dir()
    
    # 获取模型路径
    pets_model_path = input("请输入PETS模型路径（例如'pets_model_final.pth'）: ")

    # 检查文件是否存在
    if not os.path.exists(pets_model_path):
        print(f"错误: 找不到PETS模型文件 '{pets_model_path}'")
        return
    
    # 是否渲染
    render_mode = input("是否渲染环境? (y/n): ").lower() == 'y'
    
    # 评估次数
    try:
        num_episodes = int(input("评估次数 (默认为5): ") or "5")
    except ValueError:
        print("输入无效，使用默认值5")
        num_episodes = 5
    
    # 评估PETS模型
    pets_results = evaluate_pets_model(
        model_path=pets_model_path,
        env_name="Hopper-v5",
        num_episodes=num_episodes,
        render=render_mode
    )
    
    # 保存结果
    results = {
        "pets": {
            "model_path": pets_model_path,
            "mean_return": float(pets_results["mean_return"]),
            "std_return": float(pets_results["std_return"]),
            "returns": [float(r) for r in pets_results["returns"]],
            "num_episodes": num_episodes
        }
    }
    
    with open(f"{eval_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # 绘制评估结果
    plt.figure(figsize=(10, 6))
    
    # 展示每个episode的回报
    plt.bar(range(1, num_episodes+1), pets_results["returns"], color='blue', alpha=0.7)
    
    plt.axhline(y=pets_results["mean_return"], color='red', linestyle='--', alpha=0.7, 
                label=f'平均值: {pets_results["mean_return"]:.2f}')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'PETS 模型评估结果 (平均: {pets_results["mean_return"]:.2f} ± {pets_results["std_return"]:.2f})')
    plt.xticks(range(1, num_episodes+1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/evaluation_results.png", dpi=300)
    plt.show()
    
    print(f"评估结果已保存到 {eval_dir}")

def plot_training_returns(return_list_path):
    """绘制训练过程中的回报曲线"""
    # 读取回报数据
    returns = []
    with open(return_list_path, 'r') as f:
        for i, line in enumerate(f):
            episode, return_value = line.strip().split(', ')
            returns.append(float(return_value))
            if i > 105:
                break   
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制回报曲线
    plt.plot(range(1, len(returns) + 1), returns, 'b-', label='Episode Return')
    
    # # 计算移动平均（窗口大小为10）
    # window_size = 10
    # moving_avg = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
    # plt.plot(range(window_size, len(returns) + 1), moving_avg, 'r-', label=f'{window_size}-Episode Moving Average')
    
    # 添加网格和标签
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Returns over 100 Episodes')
    plt.legend()
    
    # 保存图像
    plt.savefig('experiments/PETS_20250423_102608/training_returns.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 绘制训练回报曲线
    # plot_training_returns('experiments/PETS_20250423_102608/return_list.txt')
    # 比较模型
    compare_models() 