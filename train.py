import torch
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.ppo_mask_recurrent.policies import MRActorCriticPolicy
from alphagen.rl.ppo_mask_recurrent.ppo_mask_recurrent import MRPPO
from alphagen.utils.random import reseed_everything

if __name__ == '__main__':
    from alphagen.data.evaluation import QLibEvaluation
    from alphagen.data.prebuilt_exprs import TARGET_20D

    SEED = 0
    reseed_everything(SEED)

    device = torch.device('cpu')

    ev = QLibEvaluation('csi300', '2018-01-01', '2018-12-31', TARGET_20D, device=device)
    env = AlphaEnv(ev=ev, dirty_action=True)

    callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix=f'maskable_ppo_seed{SEED}')

    # model = MaskablePPO(
    #     policy='MlpPolicy',
    #     env=env,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     clip_range_vf=None,
    #     normalize_advantage=True,
    #     ent_coef=0.0,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     target_kl=None,
    #     tensorboard_log=None,
    #     create_eval_env=False,
    #     policy_kwargs={},
    #     verbose=1,
    #     seed=SEED,
    #     device=device,
    #     _init_setup_model=True,
    # )
    model = MaskablePPO(
        policy='MlpPolicy',
        env=env,
        verbose=1
    )
    model.learn(total_timesteps=1000000, callback=callback)
