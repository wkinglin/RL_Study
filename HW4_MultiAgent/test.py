from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import parallel_to_aec
env = simple_spread_v3.parallel_env(render_mode="rgb_array")
env.reset(seed=42)

# for i, agent in enumerate(env.agent_iter()):
#     observation, reward, termination, truncation, info = env.last(agent)
#     observation1, reward1, termination1, truncation1, info1 = env.last()
#     print(agent)
#     print(observation)
#     print(observation1)
#     print(info)
    
#     if i>10 : break


#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()

#     env.step(action)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)

    print(observations)
    print
    

    env = parallel_to_aec(env)
    print(env.last("agent_0"))
    print(env.last("agent_1"))
    print(env.last("agent_2"))

    break

env.close()