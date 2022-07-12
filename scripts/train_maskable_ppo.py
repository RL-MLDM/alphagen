import torch

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.random import reseed_everything

if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device("cuda:0")
    env = AlphaEnv("csi100", "2018-01-01", "2018-12-31", device)

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=1000000)
