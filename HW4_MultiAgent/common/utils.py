import inspect
import functools
import torch
import numpy as np
import matplotlib.pyplot as plt


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
    # q_targets.shape = (episode_num, max_episode_len， n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. terminated用来将超出当前episode长度的q_targets和r置为0
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len，n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return

def load_npy_data(path):
    data = np.load(path)
    return data

def draw_plt(num, win_rates, episode_rewards): 
        evaluate_cycle = 5000

        plt.figure(figsize=(10, 8), dpi=300)  # 设置更大的图形尺寸
        
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(win_rates)), win_rates)
        # plt.xlabel('timesteps {}'.format(evaluate_cycle), labelpad=10)  # 增加标签与轴的距离
        # plt.ylabel('win_rates', labelpad=10)
        # plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
        # plt.tick_params(axis='both', which='major', labelsize=10)  # 设置刻度标签大小

        plt.subplot(2, 1, 2)
        x_ticks = [step * 3 for step in range(len(episode_rewards))]
        plt.plot(x_ticks, episode_rewards)
        # plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('timesteps {}'.format(evaluate_cycle), labelpad=10)
        plt.ylabel('episode_rewards', labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout(pad=2.0)  # 增加子图之间的间距

        plt.savefig("../result/vdn/5m_vs_6m/" + '/plt_{}.png'.format(num), format='png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    rewards_data = load_npy_data("../result/vdn/5m_vs_6m/episode_rewards_0.npy")
    win_rates_data = load_npy_data("../result/vdn/5m_vs_6m/win_rates_0.npy")
    print(rewards_data)
    print(win_rates_data)
    draw_plt(1, win_rates_data, rewards_data)
    