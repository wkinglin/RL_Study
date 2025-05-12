# run_mbpo_hopper.py
import gymnasium as gym
import torch, numpy as np
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
from mbrl.models import Ensemble
from mbrl.env.termination_fns import hopper
import os
import hydra
from omegaconf import DictConfig
import mbrl.util.env

os.environ["GYM_MUJOCO_BACKEND"] = "mujoco"

@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    # 创建环境
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)

    # 设置随机种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    return mbpo.train(env, test_env, term_fn, cfg)

if __name__ == "__main__":
    run()
