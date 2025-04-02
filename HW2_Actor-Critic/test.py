import gymnasium as gym

def run_random_agent(env_name="HalfCheetah-v4", num_episodes=5):
    env = gym.make(env_name, render_mode="human")  # 需要 GUI 环境支持
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action = env.action_space.sample()  # 采取随机动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}")
    
    env.close()

if __name__ == "__main__":
    run_random_agent()