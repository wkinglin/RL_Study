import gymnasium as gym

env = gym.make("HalfCheetah - v5")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

for episode in range (num_episodes) :
    obs , _ = env.reset ()
    done = False
    while not done:
        action = my_policy( obs )
        next_obs , reward , done , _ , _ = env . step ( action )
        my_buffer.push ( obs , next_obs , action , reward , done )
        obs = next_obs
    batch = my_buffer.sample(batch_size)
    my_policy.train(batch)