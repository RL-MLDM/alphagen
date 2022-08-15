import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from alphagen.data.evaluation import QLibEvaluation
from alphagen.data.prebuilt_exprs import TARGET_20D
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.random import reseed_everything

if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device('cpu')
    ev = QLibEvaluation('csi300', '2018-01-01', '2018-12-31', TARGET_20D, device=device)
    env = AlphaEnv(ev=ev, dirty_action=True, print_expr=True)

    STEP = 30000
    model = MaskablePPO.load(f'logs/maskable_ppo_seed0_{STEP}_steps')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(mean_reward, std_reward)
