import torch

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.ppo_mask_recurrent.policies import MRActorCriticPolicy
from alphagen.rl.ppo_mask_recurrent.ppo_mask_recurrent import MRPPO
from alphagen.utils.random import reseed_everything


if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device("cuda:0")
    env = AlphaEnv("csi100", "2018-01-01", "2018-12-31", device)
    eval_env = AlphaEnv("csi100", "2018-01-01", "2018-12-31", device, print_expr=True)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='maskable_ppo')

    # model = MRPPO(MRActorCriticPolicy, env, verbose=1)
    model = MaskablePPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
