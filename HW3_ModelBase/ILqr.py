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
    
    def backward_pass(self, x_seq, u_seq, A_seq, B_seq, target_state=None):
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
            q_t, r_t, q_x_t, r_u_t = self.quadratize_cost(x_t, u_t, target_state)
            
            # 计算Q函数的二次近似参数
            Q_x = q_x_t + np.dot(V_x.T, A_t)
            Q_u = r_u_t + np.dot(V_x.T, B_t)
            Q_xx = q_t + np.dot(A_t.T, np.dot(V_xx, A_t))
            # Q_xu = np.dot(A_t.T, np.dot(V_xx, B_t))
            Q_ux = np.dot(B_t.T, np.dot(V_xx, A_t))
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

        total_reward = np.sum([reward_fn(x_seq[t], u_seq[t]) for t in range(self.planning_horizon)])

        # iLQR迭代
        for it in range(self.max_iter):
            # 计算A B矩阵
            A_seq = []
            B_seq = []
            for t in range(self.planning_horizon):
                A_t, B_t = self.linearize_dynamics(dynamics_fn, x_seq[t], u_seq[t])
                A_seq.append(A_t)
                B_seq.append(B_t)

            # 计算A B矩阵 
            # 向后传递
            k_seq, K_seq = self.backward_pass(x_seq, u_seq, A_seq, B_seq, target_state)
            
            # 线性搜索找到最佳的alpha值
            alpha = 1.0
            accept = False
            
            for _ in range(5):  # 最多尝试5个alpha值
                # 前向传递
                x_seq_new, u_seq_new = self.forward_pass(current_state, u_seq, k_seq, K_seq, dynamics_fn, alpha)
                
                # 计算新轨迹的总代价
                new_cost = np.sum([self.cost_function(x_seq_new[t], u_seq_new[t], target_state) for t in range(self.planning_horizon)])
                new_cost += self.terminal_cost(x_seq_new[-1], target_state)

                print(f"new_cost: {new_cost}, total_cost: {total_cost}")
                
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