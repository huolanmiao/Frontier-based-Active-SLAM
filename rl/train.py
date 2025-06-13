import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributions.normal
from typing import List, Optional, Tuple

from AgentPPO import AgentDiscretePPO
from config import Config
from run import train_agent, valid_agent

from ActiveSLAMEnv import ActiveSLAMEnv
ARY = np.ndarray
TEN = th.Tensor


def train_ppo_for_ASLAM(gpu_id: int = 0):
    agent_class = AgentDiscretePPO  # DRL algorithm name
    env_class = ActiveSLAMEnv  # Environment name, see `erl_envs.py` for more environments
    
    
    env_args = {
        'env_name': 'ActiveSLAMEnv',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': [432,455],  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 2,  # the torque applied to free end of the pendulum
        'if_discrete': True,  # continuous action space, symbols → direction, value → force
        
        'vector_file_path': "./24D_196560_1.npy",
    }
    

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e20)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small

    args.gpu_id = gpu_id
    train_agent(args, if_single_process=True)  # train an agent
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)




if __name__ == "__main__":
    train_ppo_for_ASLAM()
    
    # tensorboard --logdir=./rl_topk_agent/TreeSearchEnv_DiscretePPO_0/tensorboard