import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO

from alphagen.rl.ppo_mask_recurrent.policies import MRActorCriticPolicy
from alphagen.rl.ppo_mask_recurrent.ppo_mask_recurrent import MRPPO

import gym

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    env = ActionMasker(env, lambda env: np.ones(env.action_space.n, dtype=bool))
    model = MRPPO(MRActorCriticPolicy, env, verbose=1)
    model.learn(50000)
