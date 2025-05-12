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

class DynamicsModel(nn.Module):
    """基于神经网络的动力学模型"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, device=None):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 状态转移网络 - 使用更深的网络结构
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(self.device)
        
        # 奖励预测网络 - 添加更多层
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state, action):
        """前向传播，预测下一个状态和奖励"""
        # 确保输入在正确的设备上
        if not state.is_cuda and self.device.type == 'cuda':
            state = state.to(self.device)
        if not action.is_cuda and self.device.type == 'cuda':
            action = action.to(self.device)
            
        x = torch.cat([state, action], dim=-1)
        next_state_delta = self.dynamics_net(x)
        next_state = state + next_state_delta  # 预测状态变化量
        reward = self.reward_net(x)
        return next_state, reward
    
    def predict_next_state(self, state, action):
        """预测下一个状态"""
        # 确保输入在正确的设备上
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_state, _ = self.forward(state_tensor, action_tensor)
        return next_state.squeeze(0).cpu().numpy()
    
    def predict_reward(self, state, action):
        """预测奖励"""
        # 确保输入在正确的设备上
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, reward = self.forward(state_tensor, action_tensor)
        return reward.item()
        
    def compute_rollout_loss(self, initial_states, actions, target_states, rollout_steps=3):
        """计算多步rollout的损失
        
        参数:
            initial_states: 初始状态张量 [batch_size, state_dim]
            actions: 动作序列张量 [batch_size, rollout_steps, action_dim]
            target_states: 目标状态序列张量 [batch_size, rollout_steps, state_dim]
            rollout_steps: rollout的步数
            
        返回:
            rollout_loss: 多步预测的损失
        """
        current_states = initial_states
        rollout_loss = 0
        
        for step in range(rollout_steps):
            # 获取当前步骤的动作
            current_actions = actions[:, step]
            
            # 预测下一个状态
            next_states, _ = self.forward(current_states, current_actions)
            
            # 计算当前步骤的损失
            step_loss = F.mse_loss(next_states, target_states[:, step])
            rollout_loss += step_loss
            
            # 更新当前状态
            current_states = next_states
            
        return rollout_loss / rollout_steps

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
        
        # 代价函数参数 - 调整权重以鼓励向前移动
        self.Q = np.eye(num_states) * 0.1     # 状态代价矩阵 - 降低权重
        self.R = np.eye(num_actions) * 0.01   # 控制代价矩阵 - 降低权重
        self.Qf = np.eye(num_states) * 1.0    # 终端状态代价矩阵
        
        # 特别关注x位置和速度
        self.Q[0, 0] = 0.0  # x位置权重
        self.Q[5, 5] = 1.0  # x速度权重
        self.Qf[0, 0] = 0.0  # 终端x位置权重
        self.Qf[5, 5] = 10.0  # 终端x速度权重
        
        # 动作边界
        self.action_low = action_low
        self.action_high = action_high
    
    def cost_function(self, state, action, next_state, next_action, reward_fn=None):
        """计算状态-动作对的代价"""
        next_state_reward = reward_fn(next_state, next_action)
        current_reward = reward_fn(state, action)
        cost = current_reward - next_state_reward
        return cost

    def terminal_cost(self, state, target_state=None, reward_fn=None):
        """计算终端状态的代价"""
        if target_state is None:
            return 0
            
        # 终端代价
        action = np.zeros(self.num_actions)
        
        target_reward = reward_fn(target_state, action)
        current_reward = reward_fn(state, action)
        terminal_cost = current_reward - target_reward 
        
        return terminal_cost
    
    def quadratize_cost(self, x, u, target_state=None, reward_fn=None):
        """二次化状态-动作代价函数"""
        # 定义求导函数
        def grad_x(f, x, u, h=1e-4):
            """计算函数f关于x的梯度"""
            n = len(x)
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                grad[i] = (f(x_plus, u) - f(x_minus, u)) / (2 * h)
            return grad
            
        def grad_u(f, x, u, h=1e-4):
            """计算函数f关于u的梯度"""
            m = len(u)
            grad = np.zeros(m)
            for i in range(m):
                u_plus = u.copy()
                u_minus = u.copy()
                u_plus[i] += h
                u_minus[i] -= h
                grad[i] = (f(x, u_plus) - f(x, u_minus)) / (2 * h)
            return grad
            
        def hess_xx(f, x, u, h=1e-4):
            """计算函数f关于x的Hessian矩阵"""
            n = len(x)
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    
                    x_pm[i] += h
                    x_pm[j] -= h
                    
                    x_mp[i] -= h
                    x_mp[j] += h
                    
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    H[i, j] = (f(x_pp, u) - f(x_pm, u) - f(x_mp, u) + f(x_mm, u)) / (4 * h * h)
            return H
            
        def hess_uu(f, x, u, h=1e-4):
            """计算函数f关于u的Hessian矩阵"""
            m = len(u)
            H = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    u_pp = u.copy()
                    u_pm = u.copy()
                    u_mp = u.copy()
                    u_mm = u.copy()
                    
                    u_pp[i] += h
                    u_pp[j] += h
                    
                    u_pm[i] += h
                    u_pm[j] -= h
                    
                    u_mp[i] -= h
                    u_mp[j] += h
                    
                    u_mm[i] -= h
                    u_mm[j] -= h
                    
                    H[i, j] = (f(x, u_pp) - f(x, u_pm) - f(x, u_mp) + f(x, u_mm)) / (4 * h * h)
            return H
            
        def hess_xu(f, x, u, h=1e-4):
            """计算函数f关于x和u的交叉Hessian矩阵"""
            n = len(x)
            m = len(u)
            H = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    x_p = x.copy()
                    x_m = x.copy()
                    u_p = u.copy()
                    u_m = u.copy()
                    
                    x_p[i] += h
                    x_m[i] -= h
                    u_p[j] += h
                    u_m[j] -= h
                    
                    H[i, j] = (f(x_p, u_p) - f(x_p, u_m) - f(x_m, u_p) + f(x_m, u_m)) / (4 * h * h)
            return H
        
        # 代价函数 (负的奖励函数)
        if reward_fn is not None:
            cost_fn = lambda s, a: -reward_fn(s, a)
            
            # 计算一阶导数（梯度）
            q_x = grad_x(cost_fn, x, u)  # 代价关于状态的梯度
            r_u = grad_u(cost_fn, x, u)  # 代价关于动作的梯度
            
            # 计算二阶导数（Hessian）
            q = hess_xx(cost_fn, x, u)    # 代价关于状态的Hessian
            r = hess_uu(cost_fn, x, u)    # 代价关于动作的Hessian
            cross = hess_xu(cost_fn, x, u)  # 代价关于状态和动作的交叉Hessian
            
            # 确保Hessian矩阵正定（iLQR要求）
            # 添加小的正则化项
            min_eig_q = np.min(np.linalg.eigvals(q))
            min_eig_r = np.min(np.linalg.eigvals(r))
            
            if min_eig_q < 0:
                q += np.eye(len(x)) * (abs(min_eig_q) + 1e-4)
            if min_eig_r < 0:
                r += np.eye(len(u)) * (abs(min_eig_r) + 1e-4)
                
            return q, r, q_x, r_u, cross.T  # 返回交叉项的转置，使其符合l_ux的方向
        else:
            # 如果没有reward_fn，使用预定义的Q和R矩阵
            q = self.Q if target_state is not None else np.zeros_like(self.Q)
            r = self.R
            
            # 代价函数的一阶导数(梯度)
            if target_state is not None:
                q_x = np.dot(self.Q, x - target_state)
            else:
                q_x = np.zeros(self.num_states)
                
            r_u = np.dot(self.R, u)
            
            # 交叉项为零
            cross = np.zeros((self.num_actions, self.num_states))
            
            return q, r, q_x, r_u, cross
    
    def backward_pass(self, x_seq, u_seq, A_seq, B_seq, target_state=None, reward_fn=None):
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
            A_t = A_seq[t]
            B_t = B_seq[t]
            
            # 获取二次化代价函数
            q_t, r_t, q_x_t, r_u_t, cross_t = self.quadratize_cost(x_t, u_t, target_state, reward_fn)
            
            # 计算Q函数的二次近似参数
            Q_x = q_x_t + np.dot(A_t.T, V_x)
            Q_u = r_u_t + np.dot(B_t.T, V_x)
            Q_xx = q_t + np.dot(A_t.T, np.dot(V_xx, A_t))
            Q_ux = cross_t + np.dot(B_t.T, np.dot(V_xx, A_t))
            Q_uu = r_t + np.dot(B_t.T, np.dot(V_xx, B_t))
            
            
            # 添加正则化项以确保Q_uu是正定的
            Q_uu_reg = Q_uu + np.eye(self.num_actions) * self.reg_factor
            
            # 计算反馈增益K和前馈增益k
            k_t = -np.linalg.solve(Q_uu_reg, Q_u)
            K_t = -np.linalg.solve(Q_uu_reg, Q_ux)
            
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
    
    def linearize_dynamics(self, dynamics_fn, state, action):
        """
        在某个状态(state)和动作(action)点线性化 dynamics_fn，返回 A 和 B 矩阵
        """
        # 确保输入是 PyTorch 张量
        state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
        action = torch.tensor(action, dtype=torch.float32, requires_grad=True)

        # 定义一个包装函数，确保返回 PyTorch 张量
        def dynamics_fn_wrapper(s, a):
            # 调用原始动力学函数
            next_state = dynamics_fn(s.detach().numpy(), a.detach().numpy())
            # 将结果转换回 PyTorch 张量
            return torch.tensor(next_state, dtype=torch.float32)

        # 对 state 求雅可比 A
        A = torch.autograd.functional.jacobian(
            lambda s: dynamics_fn_wrapper(s, action), 
            state
        )
        
        # 对 action 求雅可比 B
        B = torch.autograd.functional.jacobian(
            lambda a: dynamics_fn_wrapper(state, a), 
            action
        )
        
        return A.detach().numpy(), B.detach().numpy()


    def plan(self, current_state, dynamics_fn, reward_fn, target_state=None, init_u_seq=None):
        """使用iLQR算法进行规划
        
        参数:
            current_state: 当前状态
            dynamics_fn: 动力学模型函数，接收(state, action)返回next_state
            reward_fn: 奖励模型函数，接收(state, action)返回reward
            target_state: 目标状态，可选
            init_u_seq: 初始控制序列，可选
        
        返回:
            最优动作（控制序列的第一个动作）
        """
        # 初始化轨迹
        # u_seq: 控制序列
        if init_u_seq is None:
            u_seq = [np.zeros(self.num_actions) for _ in range(self.planning_horizon)]
        else:
            u_seq = init_u_seq
            
        # x_seq: 状态序列
        x_seq = [np.zeros(self.num_states) for _ in range(self.planning_horizon + 1)]
        x_seq[0] = current_state
        
        # 使用动力学模型前向预测轨迹
        for t in range(self.planning_horizon):
            x_seq[t+1] = dynamics_fn(x_seq[t], u_seq[t])

        total_cost = np.sum([self.cost_function(x_seq[t], u_seq[t], x_seq[t+1], u_seq[t+1], reward_fn) for t in range(self.planning_horizon-1)])
        total_cost += self.terminal_cost(x_seq[-1], target_state, reward_fn)

        # iLQR迭代
        for it in range(self.max_iter):
            # 计算A B矩阵
            A_seq = []
            B_seq = []
            for t in range(self.planning_horizon):
                A_t, B_t = self.linearize_dynamics(dynamics_fn, x_seq[t], u_seq[t])
                A_seq.append(A_t)
                B_seq.append(B_t)

            # 向后传递
            k_seq, K_seq = self.backward_pass(x_seq, u_seq, A_seq, B_seq, target_state, reward_fn)
            
            # 线性搜索找到最佳的alpha值
            alpha = 1.0
            accept = False
            
            for _ in range(5):  # 最多尝试5个alpha值
                # 前向传递
                x_seq_new, u_seq_new = self.forward_pass(current_state, u_seq, k_seq, K_seq, dynamics_fn, alpha)
                
                # 计算新轨迹的总代价
                new_cost = np.sum([self.cost_function(x_seq_new[t], u_seq_new[t], x_seq_new[t+1], u_seq_new[t+1], reward_fn) for t in range(self.planning_horizon-1)])
                new_cost += self.terminal_cost(x_seq_new[-1], target_state, reward_fn)

                # print(f"new_cost: {new_cost}, total_cost: {total_cost}")
                
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

class ModelBasedRL_NN:
    def __init__(self, num_states, num_actions, planning_horizon=10, model_update_frequency=10, env=None,
                 hidden_dim=256, lr=1e-3, batch_size=64, train_epochs=5):
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型并移动到设备
        self.model = DynamicsModel(num_states, num_actions, hidden_dim, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 经验回放
        self.experience = []
        
        # 创建iLQR规划器
        self.planner = iLQRPlanner(
            num_states=num_states,
            num_actions=num_actions,
            planning_horizon=planning_horizon,
            max_iter=100,
            convergence_threshold=1e-5,
            action_low=self.action_low,
            action_high=self.action_high
        )
        
    def train_dynamics_model(self):
        """使用梯度下降训练神经网络动力学模型"""
        if len(self.experience) < 50:
            return False
            
        # 准备训练数据并确保在正确的设备上
        states, actions, next_states, rewards = [], [], [], []
        for s, a, ns, r in self.experience:
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            
        # 转换为Tensor并移动到设备
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
                
                # 计算单步损失
                state_loss = F.mse_loss(pred_next_states, batch_next_states)
                reward_loss = F.mse_loss(pred_rewards, batch_rewards)
                
                # 总损失 = 单步损失 + rollout损失
                loss = state_loss + reward_loss

                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
    
    def plan(self, current_state, target_state=None):
        """使用iLQR或随机采样方法进行规划"""
        # 如果模型未训练充分，返回随机动作
        if len(self.experience) < 50:
            return np.random.uniform(self.action_low, self.action_high)
        
        # 优先使用iLQR (如果线性近似模型可用)
        dynamics_fn = lambda s, a: self.predict_next_state(s, a)
        reward_fn = lambda s, a: self.predict_reward(s, a)
        best_action = self.planner.plan(
            current_state=current_state,
            dynamics_fn=dynamics_fn,
            reward_fn=reward_fn,
            target_state=target_state
        )
        
        return best_action

    def run_episode(self, env, num_steps=500):
        """运行一个完整的episode"""
        current_state, _ = env.reset()  # Gymnasium环境的reset方法返回(observation, info)
        total_reward = 0
        
        # 尝试训练初始模型
        self.train_dynamics_model()
        
        # 设置目标状态
        target_state = np.zeros_like(current_state)
        # 设置目标高度（z坐标）
        target_state[0] = 1.4  # 目标高度
        # 设置目标角度（躯干直立）
        target_state[1] = 0.0  # 目标角度
        # 设置目标速度（向前移动）
        target_state[5] = 2.0  # 目标x方向速度
        # 设置目标角速度（保持稳定）
        target_state[7] = 0.0  # 目标躯干角速度
        
        for step in range(num_steps):
            # 1. 规划动作
            action = self.plan(current_state, target_state)
            
            # 2. 执行动作并观察结果
            next_state, reward, terminated, truncated, _ = env.step(action)  # Gymnasium环境的step方法返回5个值
            done = terminated or truncated
            total_reward += reward
            
            # 3. 存储经验
            self.experience.append((current_state, action, next_state, reward))
            
            # 限制经验回放的大小以避免内存问题
            if len(self.experience) > 50000:
                self.experience = self.experience[-5000:]
                print(f"Experience size: {len(self.experience)}")
            
            # 4. 周期性地训练模型
            if step % self.model_update_frequency == 0:
                self.train_dynamics_model()
                
            # 偶尔打印进度
            if step % 50 == 0:
                print(f"Step {step}, Cumulative Reward: {total_reward}")
                print(f"Current x position: {current_state[0]:.2f}, Target x position: {target_state[0]:.2f}")
                print(f"Current x velocity: {current_state[5]:.2f}, Target x velocity: {target_state[5]:.2f}")
                
            current_state = next_state
            if done:
                print(f"Total Step: {step}")
                break
                
        return total_reward

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 创建Hopper环境
    env = gym.make("Hopper-v5", 
                #    render_mode="human", 
                   terminate_when_unhealthy=True, 
                   forward_reward_weight = 10)
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
        planning_horizon=20,              # 规划步数
        model_update_frequency=10,       # 每10步更新一次模型
        env=env,
        hidden_dim=256,                  # 隐藏层维度
        lr=1e-3,                         # 学习率
        batch_size=64,                   # 批次大小
        train_epochs=10                   # 每次更新后，dynamic model训练轮数
    )
    
    # 训练过程
    num_episodes = 30
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