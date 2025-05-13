import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False, seed=42):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []  
        # Reset environment
        init_obs, _ = self.env.reset(seed=seed)
        
         # 修改
        observations = init_obs
        terminated = False
        truncated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':   
            state = self.env.state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        # 修改
        while not (terminated or truncated) and step < self.episode_limit:
            global_state = self.env.state()

            actions = {}
            actions_list = []
            avail_actions = []
            actions_onehot = []       

            # Choose actions for each agent
            for i, agent in enumerate(self.env.agents):   
                # In PettingZoo, all actions are available (no mask)
                # Create a dummy avail_action vector of all ones
                avail_action = np.ones(self.n_actions)
                
                # 不确定是否应该sample
                if self.args.alg == 'maven':
                    action = self.agents.choose_action(observations[agent], last_action[i], i,
                                                      avail_action, epsilon, maven_z)
                else:
                    action = self.agents.choose_action(observations[agent], last_action[i], i,
                                                      avail_action, epsilon)
                
                # Generate onehot vector of the action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                
                actions[agent] = action
                actions_list.append(np.int64(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[i] = action_onehot
            
            # 将tensor转为npint
            actions = {agent: int(action) for agent, action in actions.items()}
            obs, reward, terminated, truncated, info = self.env.step(actions)
            
             # 求reward的加和
            rewards = sum(rd for rd in reward.values())
            # terminated和truncated转为bool
            flag = False
            for item in terminated.values():
                flag = flag or item
            terminated = flag
            flag = False
            for item in truncated.values():
                flag = flag or item
            truncated = flag

            win_tag = True if (terminated or truncated) and  'battle_won' in info and info['battle_won'] else False
            
            # observations转为数组
            observations = [np.array(observations[agent]) for agent in observations.keys()]
            observations = np.array(observations)
            o.append(observations)
            s.append(global_state)
            u.append(np.reshape(actions_list, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([rewards])
            terminate.append([terminated or truncated])
            padded.append([0.])
            episode_reward += rewards
            step += 1
            observations = obs
            
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
                
        # Get final observations
        observations = obs
        global_state = self.env.state()
        
        # obs转为数组
        observations = [np.array(observations[agent]) for agent in observations.keys()]
        observations = np.array(observations)
        
        o.append(observations)
        s.append(global_state)
        
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        
        # Get available actions for the last observation
        avail_actions = []
        for _ in range(self.n_agents):
            # In PettingZoo, all actions are available (no mask)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
            
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # If step < self.episode_limit, padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                      s=s.copy(),
                      u=u.copy(),
                      r=r.copy(),
                      avail_u=avail_u.copy(),
                      o_next=o_next.copy(),
                      s_next=s_next.copy(),
                      avail_u_next=avail_u_next.copy(),
                      u_onehot=u_onehot.copy(),
                      padded=padded.copy(),
                      terminated=terminate.copy()
                      )

        # Add episode dim
        for key in episode.keys():
            try:
                episode[key] = np.array([episode[key]])
            except:
                for item in episode[key]:
                    print(item.shape)
                # from IPython import embed
                # embed()
                raise Exception('np array expand error')
            
        if not evaluate:
            self.epsilon = epsilon
            
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
            
        return episode, episode_reward, win_tag, step


# RolloutWorker for communication
# 不一定正确
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False, seed=42):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
            
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        
        # Reset environment
        init_obs, _ = self.env.reset(seed=seed)
        
        terminated = False
        truncated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            
        while not (terminated or truncated) and step < self.episode_limit:
            # Get observations for all agents
            observations = {}
            for agent in self.env.agents:
                obs, _, _, _, _ = self.env.last(agent)
                observations[agent] = obs
            
            # Construct global state (concatenation of all observations)
            global_state = np.concatenate(list(observations.values()), axis=0)
            
            # Get weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(list(observations.values())), last_action)
            
            actions = {}
            actions_list = []
            avail_actions = []
            actions_onehot = []
            
            # Choose actions for each agent
            for i, agent in enumerate(self.env.agents):
                # In PettingZoo, all actions are available (no mask)
                avail_action = np.ones(self.n_actions)
                action = self.agents.choose_action(weights[i], avail_action, epsilon)
                
                # Generate onehot vector of the action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                
                actions[agent] = action
                actions_list.append(np.int64(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[i] = action_onehot
            
            # Step the environment with the chosen actions
            self.env.step(actions)
            
            # Collect rewards, terminated and truncated flags
            rewards = 0
            terminated = False
            truncated = False
            info = {}
            
            for agent in self.env.agents:
                _, reward, term, trunc, _ = self.env.last(agent)
                rewards += reward
                terminated = terminated or term
                truncated = truncated or trunc
            
            # For win condition, we consider it a win if all agents complete the task
            # This needs to be adjusted based on your specific environment
            win_tag = True if (terminated or truncated) and rewards > 0 else False
            
            o.append(list(observations.values()))
            s.append(global_state)
            u.append(np.reshape(actions_list, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([rewards])
            terminate.append([terminated or truncated])
            padded.append([0.])
            episode_reward += rewards
            step += 1
            
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
                
        # Get final observations
        observations = {}
        for agent in self.env.agents:
            obs, _, _, _, _ = self.env.last(agent)
            observations[agent] = obs
            
        # Construct final global state
        global_state = np.concatenate(list(observations.values()), axis=0)
        
        o.append(list(observations.values()))
        s.append(global_state)
        
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        
        # Get available actions for the last observation
        avail_actions = []
        for _ in range(self.n_agents):
            # In PettingZoo, all actions are available (no mask)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
            
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # If step < self.episode_limit, padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                      s=s.copy(),
                      u=u.copy(),
                      r=r.copy(),
                      avail_u=avail_u.copy(),
                      o_next=o_next.copy(),
                      s_next=s_next.copy(),
                      avail_u_next=avail_u_next.copy(),
                      u_onehot=u_onehot.copy(),
                      padded=padded.copy(),
                      terminated=terminate.copy()
                      )
        # Add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
            
        if not evaluate:
            self.epsilon = epsilon
            
        return episode, episode_reward, win_tag, step
