import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.random import reseed_everything

if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device("cuda:0")
    env = AlphaEnv("csi100", "2018-01-01", "2018-12-31", device, print_expr=True)

    model = MaskablePPO.load('logs/maskable_ppo_640000_steps')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(mean_reward, std_reward)
