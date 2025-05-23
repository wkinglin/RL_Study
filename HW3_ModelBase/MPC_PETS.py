import numpy as np
from scipy.stats import truncnorm
import gymnasium as gym
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt
import json
import os
import time


class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound,
                 lower_bound):
        self.n_sequence = n_sequence
        self.elite_ratio = elite_ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fake_env = fake_env

    def optimize(self, state, init_mean, init_var):
        mean, var = init_mean, init_var
        self.planning_horizon = len(mean) // 3  # 计算规划步数
        state = np.tile(state, (self.n_sequence, 1))

        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)),
                var)
            
            # 生成动作序列
            action_sequences = []
            for _ in range(self.n_sequence):
                # 为每个序列生成动作
                sequence = []
                for i in range(self.planning_horizon):
                    # 为每一步生成3维动作
                    idx = i * 3
                    action_mean = mean[idx:idx+3]
                    action_var = constrained_var[idx:idx+3]
                    action = truncnorm.rvs(
                        -2, 2, 
                        loc=action_mean, 
                        scale=np.sqrt(action_var)
                    )
                    sequence.append(action)
                action_sequences.append(sequence)
            
            # 转换为numpy数组，形状为(n_sequence, planning_horizon, 3)
            action_sequences = np.array(action_sequences)
            
            # 计算每条动作序列的累积奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            # 选取累积奖励最高的若干条动作序列
            elites = action_sequences[np.argsort(
                returns)][-int(self.elite_ratio * self.n_sequence):]
            
            # 计算新的均值和方差
            new_mean = np.mean(elites, axis=0)  # (planning_horizon, 3)
            new_var = np.var(elites, axis=0)    # (planning_horizon, 3)
            
            # 展平为一维数组以匹配mean和var的形状
            new_mean = new_mean.flatten()  # (planning_horizon*3,)
            new_var = new_var.flatten()    # (planning_horizon*3,)
            
            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        # 返回最优动作序列，重塑为(planning_horizon, 3)
        return mean.reshape(self.planning_horizon, 3)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


class Swish(nn.Module):
    ''' Swish激活函数 '''
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def init_weights(m):
    ''' 初始化模型权重 '''
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond,
                torch.nn.init.normal_(torch.ones(t.shape, device=device),
                                      mean=mean,
                                      std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, FCLayer):
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    ''' 集成之后的全连接层 '''
    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim).to(device))
        self._activation = activation
        self.bias = nn.Parameter(
            torch.Tensor(ensemble_size, output_dim).to(device))

    def forward(self, x):
        return self._activation(
            torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))
    
class EnsembleModel(nn.Module):
    ''' 环境模型集成 '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 ensemble_size=5,
                 learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        # 输出包括均值和方差,因此是状态与奖励维度之和的两倍
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter((torch.ones(
            (1, self._output_dim // 2)).float() / 2).to(device),
                                        requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones(
            (1, self._output_dim // 2)).float() * 10).to(device),
                                        requires_grad=False)

        # 确保输入维度正确
        input_dim = state_dim + action_dim
        self.layer1 = FCLayer(input_dim, 200, ensemble_size, Swish())
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size,
                              nn.Identity())
        self.apply(init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(
            self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        # 在PETS算法中,将方差控制在最小值和最大值之间
        logvar = self._max_logvar - F.softplus(
            self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) *
                                             inverse_var,
                                             dim=-1),
                                  dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(
            self._min_logvar)
        loss.backward()
        self.optimizer.step()

class EnsembleDynamicsModel:
    ''' 环境模型集成,加入精细化的训练 '''
    def __init__(self, state_dim, action_dim, num_network=5):
        self._num_network = num_network
        self._state_dim, self._action_dim = state_dim, action_dim
        self.model = EnsembleModel(state_dim,
                                   action_dim,
                                   ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self,
              inputs,
              labels,
              batch_size=64,
              holdout_ratio=0.1,
              max_iter=20):
        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:
                                                num_holdout], labels[:
                                                                     num_holdout]
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat(
            [self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat(
            [self._num_network, 1, 1])

        # 保留最好的结果
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)}

        for epoch in itertools.count():
            # 定义每一个网络的训练数据
            train_index = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self._num_network)
            ])
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[:, batch_start_pos:batch_start_pos +
                                          batch_size]
                train_input = torch.from_numpy(
                    train_inputs[batch_index]).float().to(device)
                train_label = torch.from_numpy(
                    train_labels[batch_index]).float().to(device)

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(mean,
                                                    logvar,
                                                    holdout_labels,
                                                    use_var_loss=False)
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)
                if break_condition or epoch > max_iter:  # 结束训练
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > threshold:
                self._snapshots[i] = (epoch, current)
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(
                inputs[i:min(i +
                             batch_size, inputs.shape[0])]).float().to(device)
            cur_mean, cur_var = self.model(input[None, :, :].repeat(
                [self._num_network, 1, 1]),
                                           return_log_var=False)
            mean.append(cur_mean.detach().cpu().numpy())
            var.append(cur_var.detach().cpu().numpy())
        return np.hstack(mean), np.hstack(var)

class FakeEnv:
    def __init__(self, model):
        self.model = model

    def step(self, obs, act):

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs.numpy()
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice(
            [i for i in range(self.model._num_network)], size=batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[models_to_use, batch_inds]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        return rewards, next_obs

    def propagate(self, obs, actions):
        with torch.no_grad():
            obs = np.copy(obs)
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1)
            obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)
            for i in range(actions.shape[1]):
                action = actions[:, i]  # 获取当前步骤的3维动作
                rewards, next_obs = self.step(obs, action)
                total_reward += rewards
                obs = torch.as_tensor(next_obs)
            return total_reward
        
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done

class PETS:
    ''' PETS算法 '''
    def __init__(self, env, replay_buffer, n_sequence, elite_ratio,
                 plan_horizon, num_episodes):
        self._env = env
        self._env_pool = replay_buffer

        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self._cem = CEM(n_sequence, elite_ratio, self._fake_env,
                        self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes
        
        # 保存超参数
        self.hyperparams = {
            'n_sequence': n_sequence,
            'elite_ratio': elite_ratio,
            'plan_horizon': plan_horizon,
            'num_episodes': num_episodes,
            'obs_dim': obs_dim,
            'action_dim': self._action_dim,
            'upper_bound': self.upper_bound,
            'lower_bound': self.lower_bound
        }
        
    def save_model(self, path='pets_model.pth'):
        """保存模型参数和超参数"""
        save_dict = {
            'model_state': self._model.model.state_dict(),
            'hyperparams': self.hyperparams
        }
        torch.save(save_dict, path)
        print(f"模型和超参数已保存到 {path}")
        
    def load_model(self, path='pets_model.pth'):
        """加载模型参数"""
        save_dict = torch.load(path)
        self._model.model.load_state_dict(save_dict['model_state'])
        print(f"模型已从 {path} 加载")
        # 打印加载的超参数
        if 'hyperparams' in save_dict:
            print("加载的超参数:")
            for k, v in save_dict['hyperparams'].items():
                print(f"  {k}: {v}")
        return save_dict.get('hyperparams', None)

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = env_samples[3]
        inputs = np.concatenate((obs, actions), axis=-1)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels)

    def mpc(self):
        # 初始化均值和方差，确保每个动作是3维的
        mean = np.zeros(self.plan_horizon * 3)
        for i in range(self.plan_horizon):
            mean[i*3:(i+1)*3] = (self.upper_bound + self.lower_bound) / 2.0
        var = np.zeros(self.plan_horizon * 3)
        for i in range(self.plan_horizon):
            var[i*3:(i+1)*3] = np.square(self.upper_bound - self.lower_bound) / 16
        
        obs, _ = self._env.reset()
        done = False
        episode_return = 0
        while not done:
            # 获取优化后的动作序列，形状为(planning_horizon, 3)
            actions = self._cem.optimize(obs, mean, var)
            action = actions[0]  # 选取第一个动作（3维）
            
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated
          
            self._env_pool.add(obs, action, reward, next_state, done)
            obs = next_state
            episode_return += reward
            
            # 更新mean，移除已执行的动作并在末尾添加零动作
            mean = np.zeros(self.plan_horizon * 3)
            for i in range(self.plan_horizon - 1):
                mean[i*3:(i+1)*3] = actions[i+1]
            
            # 同样更新方差
            var = np.zeros(self.plan_horizon * 3)
            for i in range(self.plan_horizon):
                var[i*3:(i+1)*3] = np.square(self.upper_bound - self.lower_bound) / 16
                
        return episode_return

    def explore(self):
        obs, _ = self._env.reset()
        done = False
        episode_return = 0
        while not done:
            action = self._env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated

            self._env_pool.add(obs, action, reward, next_state, done)
            obs = next_state
            episode_return += reward
        return episode_return

    def train(self):
        return_list = []
        explore_return = self.explore()  # 先进行随机策略的探索来收集一条序列的数据
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)

        for i_episode in range(self.num_episodes - 1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print('episode: %d, return: %d' % (i_episode + 2, episode_return))
            
            # 每5个episode保存一次模型
            if (i_episode + 2) % 5 == 0:
                self.save_model(f'pets_model_ep{i_episode + 2}.pth')
                
        # 训练结束后保存最终模型
        self.save_model('pets_model_final.pth')
        return return_list
    
if __name__ == "__main__":
    buffer_size = 100000
    n_sequence = 50
    elite_ratio = 0.2
    plan_horizon = 25
    num_episodes = 200
    env_name = 'Hopper-v5'
    env = gym.make(env_name)

    # 保存实验配置
    config = {
        'buffer_size': buffer_size,
        'n_sequence': n_sequence,
        'elite_ratio': elite_ratio,
        'plan_horizon': plan_horizon,
        'num_episodes': num_episodes,
        'env_name': env_name
    }

    # 创建实验目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/PETS_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # 保存配置文件
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    replay_buffer = ReplayBuffer(buffer_size)
    pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon,
                num_episodes)
    return_list = pets.train()

    # 保存回报列表
    with open(f"{exp_dir}/return_list.txt", 'w') as f:
        for i, ret in enumerate(return_list):
            f.write(f"{i+1}, {ret}\n")

    # 绘图并保存
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'PETS on {env_name}')
    plt.grid(True)
    plt.savefig(f"{exp_dir}/learning_curve.png", dpi=300)
    plt.show()