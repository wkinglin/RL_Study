from pettingzoo.mpe import simple_spread_v3
from common.arguments import get_common_args, get_coma_args, get_centralv_args, get_reinforce_args, get_mixer_args, get_commnet_args, get_g2anet_args
from runner_old import Runner
# from runner import Runner

if __name__ == "__main__":
    max_cycles = 30
    env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles)
    env.reset(seed=42)

    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    agents = env.agents
    print("agents:", agents)

    args.n_actions = env.action_space(agents[0]).n
    print("n_actions:", args.n_actions)

    args.n_agents = env.num_agents
    print("n_agents:", args.n_agents)

    args.state_shape = env.state().shape[0]
    print("state_shape:", args.state_shape)
    
    args.obs_shape = env.observation_space(agents[0]).shape[0]
    print("obs_shape:", args.obs_shape)
    
    # 只在单步环境中存在
    # args.episode_limit = env.max_cycles
    args.episode_limit = max_cycles

    print("episode_limit:", args.episode_limit)

    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(0)
    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))
    env.close()