import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, TensorDataset

# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
        # 添加层归一化
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 使用 Xavier 初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        torch.nn.init.constant_(self.fc_mu.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.fc_std.weight, gain=0.1)
        torch.nn.init.constant_(self.fc_std.bias, 0.0)

    def forward(self, x):
        # 检查输入中是否包含 NaN 值
        if torch.isnan(x).any():
            print(f"输入包含 NaN 值: {x}")
            raise ValueError("PolicyNet输入张量包含 NaN 值")
            
        # 对输入进行归一化
        x = (x - x.mean()) / x.std()
        
        # 通过第一个全连接层
        xx = self.fc1(x)
        
        # 添加数值稳定化
        # xx = torch.clamp(xx, min=-100, max=100)
        
        # 应用 ReLU 和层归一化
        # xx = F.relu(self.layer_norm(xx))
        xx = F.softplus(self.layer_norm(xx))
        # xx = F.softplus(xx)
        # xx = F.relu(xx)
        
        
        if torch.isnan(xx).any():
            print(f"包含 NaN 值: {xx}")
            print(f"Input x max: {x.max()}, min: {x.min()}")
            print(f"fc1 weight max: {self.fc1.weight.max()}, min: {self.fc1.weight.min()}")
            print(f"fc1 bias max: {self.fc1.bias.max()}, min: {self.fc1.bias.min()}")
            raise ValueError("PolicyNet在经过fc1后张量包含 NaN 值")

        # 计算均值和标准差
        mu = self.fc_mu(xx)
        std = F.softplus(self.fc_std(xx))
        
        # 检查中间结果是否包含 NaN 值
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print(f"中间结果包含 NaN 值 - mu: {mu}, std: {std}")
            print(f"Input x max: {x.max()}, min: {x.min()}")
            print(f"xx max: {xx.max()}, min: {xx.min()}")
            raise ValueError("PolicyNet中间计算结果包含 NaN 值")
            
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样函数
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)  # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        
        return action, log_prob

# 价值网络
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # 确保x和a的批处理维度相同
        if x.shape[0] != a.shape[0]:
            raise ValueError(f"批处理大小不匹配: 状态批大小={x.shape[0]}, 动作批大小={a.shape[0]}")
            
        # 检查并修正a的维度
        if len(a.shape) == 1:
            a = a.unsqueeze(-1)  # 添加动作维度
        elif len(a.shape) > 2:
            a = a.view(a.shape[0], -1)  # 将动作扁平化为(batch_size, action_dim)
            
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        return self.fc2(x)

class DynamicsModel(nn.Module):
    """基于神经网络的动力学模型"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态转移网络
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 奖励预测网络
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """前向传播，预测下一个状态和奖励"""
        x = torch.cat([state, action], dim=-1)
        next_state_delta = self.dynamics_net(x)
        next_state = state + next_state_delta  # 预测状态变化量
        reward = self.reward_net(x)
        return next_state, reward
    
    def predict_next_state(self, state, action):
        """预测下一个状态"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        with torch.no_grad():
            # next_state, _ = self.forward(state_tensor, action_tensor)
            next_state = self.forward(state_tensor, action_tensor)
        return next_state.squeeze(0).numpy()
    
    def predict_reward(self, state, action):
        """预测奖励"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        with torch.no_grad():
            _, reward = self.forward(state_tensor, action_tensor)
        return reward.item()

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
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        print(state.shape)
        state = np.tile(state, (self.n_sequence, 1))
        print(state.shape)

        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)),
                var)
            # 生成动作序列
            action_sequences = [X.rvs() for _ in range(self.n_sequence)
                                ] * np.sqrt(constrained_var) + mean
            # 计算每条动作序列的累积奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            # 选取累积奖励最高的若干条动作序列
            elites = action_sequences[np.argsort(
                returns)][-int(self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        return mean

class iLQRPlanner:
    """iLQR（迭代线性二次型调节器）规划器类"""
    def __init__(self, num_states, num_actions, planning_horizon=10, 
                 max_iter=10, reg_factor=1e-6, convergence_threshold=1e-4,
                 action_low=None, action_high=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        
        # iLQR特定参数
        self.max_iter = max_iter         # 最大迭代次数
        self.reg_factor = reg_factor     # 正则化因子
        self.convergence_threshold = convergence_threshold  # 收敛阈值
        
        # 代价函数参数
        self.Q = np.eye(num_states) * 1.0     # 状态代价矩阵
        self.R = np.eye(num_actions) * 0.1    # 控制代价矩阵
        self.Qf = np.eye(num_states) * 10.0   # 终端状态代价矩阵
        
        # 动作边界
        self.action_low = action_low
        self.action_high = action_high
    
    def cost_function(self, state, action, target_state=None):
        """计算状态-动作对的代价"""
        # 状态代价: (x-x_target)^T Q (x-x_target)
        state_cost = 0
        if target_state is not None:
            state_diff = state - target_state
            state_cost = np.dot(state_diff, np.dot(self.Q, state_diff))
            
        # 控制代价: u^T R u
        action_cost = np.dot(action, np.dot(self.R, action))
        
        return state_cost + action_cost
    
    def terminal_cost(self, state, target_state=None):
        """计算终端状态的代价"""
        if target_state is None:
            return 0
            
        # 终端代价: (x-x_target)^T Qf (x-x_target)
        state_diff = state - target_state
        terminal_cost = np.dot(state_diff, np.dot(self.Qf, state_diff))
        
        return terminal_cost
    
    def quadratize_cost(self, x, u, target_state=None):
        """二次化状态-动作代价函数"""
        # 代价函数的二阶导数(Hessian)
        q = self.Q if target_state is not None else np.zeros_like(self.Q)
        r = self.R
        
        # 代价函数的一阶导数(梯度)
        if target_state is not None:
            q_x = np.dot(self.Q, x - target_state)
        else:
            q_x = np.zeros(self.num_states)
            
        r_u = np.dot(self.R, u)
        
        return q, r, q_x, r_u
    
    def backward_pass(self, x_seq, u_seq, A, B, target_state=None):
        """iLQR的向后传递，计算增益矩阵K和k"""
        N = len(u_seq)  # 控制序列的长度
        
        # 初始化输出变量
        k_seq = [np.zeros(self.num_actions) for _ in range(N)]
        K_seq = [np.zeros((self.num_actions, self.num_states)) for _ in range(N)]
        
        # 初始化值函数的二次近似
        V_x = np.zeros(self.num_states)
        V_xx = np.zeros((self.num_states, self.num_states))
        
        if target_state is not None:
            # 使用终端代价初始化
            V_x = np.dot(self.Qf, x_seq[-1] - target_state)
            V_xx = self.Qf
        
        # 向后传递，从倒数第二个状态开始
        for t in range(N-1, -1, -1):
            x_t = x_seq[t]
            u_t = u_seq[t]
            
            # 获取二次化代价函数
            q_t, r_t, q_x_t, r_u_t = self.quadratize_cost(x_t, u_t, target_state)
            
            # 计算Q函数的二次近似参数
            Q_x = q_x_t + np.dot(A.T, V_x)
            Q_u = r_u_t + np.dot(B.T, V_x)
            Q_xx = q_t + np.dot(A.T, np.dot(V_xx, A))
            Q_uu = r_t + np.dot(B.T, np.dot(V_xx, B))
            Q_ux = np.dot(B.T, np.dot(V_xx, A))
            
            # 添加正则化项以确保Q_uu是正定的
            Q_uu_reg = Q_uu + np.eye(self.num_actions) * self.reg_factor
            
            # 计算反馈增益K和前馈增益k
            try:
                k_t = -np.linalg.solve(Q_uu_reg, Q_u)
                K_t = -np.linalg.solve(Q_uu_reg, Q_ux)
            except:
                # 如果无法求解，使用伪逆
                k_t = -np.dot(np.linalg.pinv(Q_uu_reg), Q_u)
                K_t = -np.dot(np.linalg.pinv(Q_uu_reg), Q_ux)
            
            k_seq[t] = k_t
            K_seq[t] = K_t
            
            # 更新值函数的二次近似
            V_x = Q_x + np.dot(K_t.T, np.dot(Q_uu, k_t)) + np.dot(K_t.T, Q_u) + np.dot(Q_ux.T, k_t)
            V_xx = Q_xx + np.dot(K_t.T, np.dot(Q_uu, K_t)) + np.dot(K_t.T, Q_ux) + np.dot(Q_ux.T, K_t)
            
            # 确保V_xx是对称的
            V_xx = (V_xx + V_xx.T) / 2
        
        return k_seq, K_seq
    
    def forward_pass(self, x_0, u_seq, k_seq, K_seq, dynamics_fn, alpha=1.0):
        """iLQR的前向传递，使用增益矩阵K和k来更新状态和控制序列"""
        N = len(u_seq)  # 控制序列的长度
        
        # 初始化新的状态和控制序列
        x_seq_new = [np.zeros(self.num_states) for _ in range(N+1)]
        u_seq_new = [np.zeros(self.num_actions) for _ in range(N)]
        
        x_seq_new[0] = x_0  # 初始状态
        
        # 前向传递，使用线性反馈控制律
        for t in range(N):
            # 计算控制修正
            delta_u = k_seq[t] + np.dot(K_seq[t], x_seq_new[t] - x_seq_new[0])
            
            # 更新控制，使用线性搜索系数alpha
            u_seq_new[t] = u_seq[t] + alpha * delta_u
            
            # 裁剪控制输入以符合动作边界
            if self.action_low is not None and self.action_high is not None:
                u_seq_new[t] = np.clip(u_seq_new[t], self.action_low, self.action_high)
            
            # 使用动力学模型预测下一个状态
            x_seq_new[t+1] = dynamics_fn(x_seq_new[t], u_seq_new[t])
        
        return x_seq_new, u_seq_new
    
    def plan(self, current_state, dynamics_fn, A, B, target_state=None, init_u_seq=None):
        """使用iLQR算法进行规划
        
        参数:
            current_state: 当前状态
            dynamics_fn: 动力学模型函数，接收(state, action)返回next_state
            A: 状态转移矩阵
            B: 控制矩阵
            target_state: 目标状态，可选
            init_u_seq: 初始控制序列，可选
        
        返回:
            最优动作（控制序列的第一个动作）
        """
        # 初始化轨迹
        if init_u_seq is None:
            u_seq = [np.zeros(self.num_actions) for _ in range(self.planning_horizon)]
        else:
            u_seq = init_u_seq
            
        x_seq = [np.zeros(self.num_states) for _ in range(self.planning_horizon + 1)]
        x_seq[0] = current_state
        
        # 使用动力学模型前向预测轨迹
        for t in range(self.planning_horizon):
            x_seq[t+1] = dynamics_fn(x_seq[t], u_seq[t])
        
        # 计算初始轨迹的总代价
        total_cost = np.sum([self.cost_function(x_seq[t], u_seq[t], target_state) for t in range(self.planning_horizon)])
        total_cost += self.terminal_cost(x_seq[-1], target_state)
        
        # iLQR迭代
        for it in range(self.max_iter):
            # 向后传递
            k_seq, K_seq = self.backward_pass(x_seq, u_seq, A, B, target_state)
            
            # 线性搜索找到最佳的alpha值
            alpha = 1.0
            accept = False
            
            for _ in range(5):  # 最多尝试5个alpha值
                # 前向传递
                x_seq_new, u_seq_new = self.forward_pass(current_state, u_seq, k_seq, K_seq, dynamics_fn, alpha)
                
                # 计算新轨迹的总代价
                new_cost = np.sum([self.cost_function(x_seq_new[t], u_seq_new[t], target_state) for t in range(self.planning_horizon)])
                new_cost += self.terminal_cost(x_seq_new[-1], target_state)
                
                # 如果代价降低，接受新轨迹
                if new_cost < total_cost:
                    x_seq = x_seq_new
                    u_seq = u_seq_new
                    total_cost = new_cost
                    accept = True
                    break
                    
                # 减小alpha进行回溯线性搜索
                alpha *= 0.5
            
            # 如果没有找到降低代价的alpha值，终止迭代
            if not accept:
                break
        
        # 返回规划的第一个动作
        return u_seq[0]

    def plan_random(self, current_state, dynamics_fn, num_samples=1000):
        """使用随机采样方法进行规划（替代iLQR方法）"""
        best_action = None
        best_total_cost = float('inf')
        
        # 生成多个随机动作序列
        for _ in range(num_samples):
            # 生成随机动作
            action = np.random.uniform(self.action_low, self.action_high)
            
            # 使用动力学模型模拟执行
            state = current_state.copy()
            total_cost = 0
            
            # 模拟这个动作下的未来几步
            for t in range(self.planning_horizon):
                # 如果是第一步，使用生成的随机动作；否则使用随机动作
                curr_action = action if t == 0 else np.random.uniform(self.action_low, self.action_high)
                
                # 计算当前步骤的代价
                total_cost += self.cost_function(state, curr_action)
                
                # 预测下一个状态
                state = dynamics_fn(state, curr_action)
            
            # 添加终端代价
            total_cost += self.terminal_cost(state)
            
            # 更新最佳动作
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_action = action
        
        return best_action

class ModelBasedRL_NN:
    def __init__(self, num_states, num_actions, planning_horizon=10, model_update_frequency=10, env=None,
                 hidden_dim=128, lr=1e-3, batch_size=64, train_epochs=5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        self.model_update_frequency = model_update_frequency
        self.env = env
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        
        # 动作空间的上下界
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        
        # 神经网络动力学模型
        # self.policy_net = PolicyNet(num_states, num_actions, hidden_dim)
        # self.q_net = QValueNet(num_states, num_actions, hidden_dim)

        self.model = DynamicsModel(num_states, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 经验回放
        self.experience = []
        
        # 线性近似动力学模型参数 (用于iLQR)
        self.A = None  # 状态转移矩阵
        self.B = None  # 控制矩阵
        self.c = None  # 偏置项
        
        # 创建iLQR规划器
        self.planner = iLQRPlanner(
            num_states=num_states,
            num_actions=num_actions,
            planning_horizon=planning_horizon,
            action_low=self.action_low,
            action_high=self.action_high
        )
        
        # 设置设备（CPU/GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_dynamics_model(self):
        """使用梯度下降训练神经网络动力学模型"""
        if len(self.experience) < 50:  # 需要足够多的数据
            return False
            
        # 准备训练数据
        states, actions, next_states, rewards = [], [], [], []
        for s, a, ns, r in self.experience:
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            
        # 转换为Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(states, actions, next_states, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 训练模型
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.train_epochs):
            epoch_loss = 0
            for batch_states, batch_actions, batch_next_states, batch_rewards in dataloader:
                self.optimizer.zero_grad()
                
                # 前向传播
                pred_next_states, pred_rewards = self.model(batch_states, batch_actions)
                # pred_next_states = self.model(batch_states, batch_actions)
                
                # 计算损失
                state_loss = F.mse_loss(pred_next_states, batch_next_states)
                reward_loss = F.mse_loss(pred_rewards, batch_rewards)
                loss = state_loss + reward_loss
                # loss = state_loss
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
        
            
        print(f"Dynamics model training loss: {total_loss / self.train_epochs:.4f}")
        return True

    def predict_next_state(self, state, action):
        """使用神经网络预测下一个状态"""
        return self.model.predict_next_state(state, action)
    
    def predict_reward(self, state, action):
        """使用神经网络预测奖励"""
        return self.model.predict_reward(state, action)
    
    def linearize_dynamics(self, state, action):
        """
        线性化神经网络动力学模型，计算 A, B 矩阵和偏置项 c。
        A应为形状(state_dim, state_dim)的矩阵
        B应为形状(state_dim, action_dim)的矩阵
        c应为形状(state_dim,)的向量
        """
        self.model.eval()
        
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32, requires_grad=True, device=self.device).unsqueeze(0)
        
        # 预测下一个状态
        next_state_pred, reward_pred = self.model(state_tensor, action_tensor)
        
        # 计算雅可比，并处理维度
        # jacobian返回形状为 [batch_out, out_dim, batch_in, in_dim]
        A_full = torch.autograd.functional.jacobian(
            lambda s: self.model(s, action_tensor),
            state_tensor
        )
        
        B_full = torch.autograd.functional.jacobian(
            lambda a: self.model(state_tensor, a),
            action_tensor
        )
        
        # 移除批次维度，得到[state_dim, state_dim]和[state_dim, action_dim]
        # PyTorch >= 1.9的处理方式
        if hasattr(A_full, 'view'):
            A = A_full.view(self.num_states, self.num_states).detach().cpu().numpy()
            B = B_full.view(self.num_states, self.num_actions).detach().cpu().numpy()
        else:
            # 处理形状为[1, state_dim, 1, state_dim]的情况
            A = A_full.squeeze(0).squeeze(1).detach().cpu().numpy()
            B = B_full.squeeze(0).squeeze(1).detach().cpu().numpy()
        
        # 打印调试信息
        # print(f"A_full原始形状: {A_full.shape}")
        # print(f"A最终形状: {A.shape}, 应为: ({self.num_states}, {self.num_states})")
        # print(f"B最终形状: {B.shape}, 应为: ({self.num_states}, {self.num_actions})")
        
        s_np = state_tensor.detach().cpu().numpy().squeeze()
        a_np = action_tensor.detach().cpu().numpy().squeeze()
        f_np = next_state_pred.detach().cpu().numpy().squeeze()
        c = f_np - A @ s_np - B @ a_np
        
        return A, B, c

    def plan(self, current_state, target_state=None):
        """使用iLQR或随机采样方法进行规划"""
        # 如果模型未训练充分，返回随机动作
        if len(self.experience) < 50:
            return np.random.uniform(self.action_low, self.action_high)
        
        # 使用当前状态下的最优动作执行线性化（更新A, B, c）
        action_guess = np.random.uniform(self.action_low, self.action_high)  # 可用策略初始化动作
        self.A, self.B, self.c = self.linearize_dynamics(current_state, action_guess)
        
        # 优先使用iLQR (如果线性近似模型可用)
        dynamics_fn = lambda s, a: self.predict_next_state(s, a)
        best_action = self.planner.plan(
            current_state=current_state,
            dynamics_fn=dynamics_fn,
            A=self.A,
            B=self.B,
            target_state=target_state
        )
        
        return best_action

    def run_episode(self, env, num_steps=500):
        """运行一个完整的episode"""
        current_state, _ = env.reset()  # Gymnasium环境的reset方法返回(observation, info)
        total_reward = 0
        
        # 尝试训练初始模型
        self.train_dynamics_model()
        
        for step in range(num_steps):
            # 1. 规划动作
            action = self.plan(current_state)
            
            # 2. 执行动作并观察结果
            next_state, reward, terminated, truncated, _ = env.step(action)  # Gymnasium环境的step方法返回5个值
            done = terminated or truncated
            total_reward += reward
            
            # 3. 存储经验
            self.experience.append((current_state, action, next_state, reward))
            
            # 限制经验回放的大小以避免内存问题
            if len(self.experience) > 10000:
                self.experience = self.experience[-5000:]
            
            # 4. 周期性地训练模型
            if step % self.model_update_frequency == 0:
                self.train_dynamics_model()
                
            # 偶尔打印进度
            if step % 50 == 0:
                print(f"Step {step}, Cumulative Reward: {total_reward}")
                
            current_state = next_state
            if done:
                print(f"Total Step: {step}")
                break
                
        return total_reward

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 创建Hopper环境
    env = gym.make("Hopper-v5")
    env.reset(seed=0)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]  # 11
    action_dim = env.action_space.shape[0]      # 3
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 创建基于神经网络的模型预测控制智能体
    agent = ModelBasedRL_NN(
        num_states=state_dim, 
        num_actions=action_dim, 
        planning_horizon=10,              # 规划步数
        model_update_frequency=10,       # 每10步更新一次模型
        env=env,
        hidden_dim=256,                  # 隐藏层维度
        lr=1e-3,                         # 学习率
        batch_size=64,                   # 批次大小
        train_epochs=10                   # 每次更新后，dynamic model训练轮数
    )
    
    # 训练过程
    num_episodes = 100
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        episode_reward = agent.run_episode(env)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} completed, Total Reward: {episode_reward}")
    
    # 打印训练结果
    print("\nTraining Completed!")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Average reward: {np.mean(episode_rewards)}")
    
    # 可视化学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes+1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Neural Network Model-Based RL Learning Curve')
    plt.grid(True)
    plt.savefig('hopper_nn_results.png')
    plt.show()