import os

import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.evaluation import LRUCache
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.random import reseed_everything


class CustomCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_checkpoint()
            self.show_top_alphas()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('cache/size', len(self.cache))

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")

    def show_top_alphas(self):
        if self.verbose > 0:
            print(self.cache.best_key, self.cache.best_value)

    @property
    def cache(self) -> LRUCache:
        return self.training_env.envs[0].unwrapped.eval.cache


if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device("cpu")
    csi100_2018 = ['SZ000001', 'SZ000002', 'SZ000063', 'SZ000069', 'SZ000166', 'SZ000333', 'SZ000538', 'SZ000625',
                   'SZ000651', 'SZ000725', 'SZ000776', 'SZ000858', 'SZ000895', 'SZ001979', 'SZ002024', 'SZ002027',
                   'SZ002142', 'SZ002252', 'SZ002304', 'SZ002352', 'SZ002415', 'SZ002558', 'SZ002594', 'SZ002736',
                   'SZ002739', 'SZ002797', 'SZ300059', 'SH600000', 'SH600010', 'SH600011', 'SH600015', 'SH600016',
                   'SH600018', 'SH600019', 'SH600023', 'SH600028', 'SH600030', 'SH600036', 'SH600048', 'SH600050',
                   'SH600061', 'SH600104', 'SH600115', 'SH600276', 'SH600309', 'SH600340', 'SH600518', 'SH600519',
                   'SH600585', 'SH600606', 'SH600637', 'SH600663', 'SH600690', 'SH600703', 'SH600795', 'SH600837',
                   'SH600887', 'SH600893', 'SH600900', 'SH600919', 'SH600958', 'SH600999', 'SH601006', 'SH601009',
                   'SH601018', 'SH601088', 'SH601111', 'SH601166', 'SH601169', 'SH601186', 'SH601211', 'SH601225',
                   'SH601229', 'SH601288', 'SH601318', 'SH601328', 'SH601336', 'SH601390', 'SH601398', 'SH601601',
                   'SH601618', 'SH601628', 'SH601633', 'SH601668', 'SH601669', 'SH601688', 'SH601727', 'SH601766',
                   'SH601788', 'SH601800', 'SH601818', 'SH601857', 'SH601881', 'SH601899', 'SH601901', 'SH601985',
                   'SH601988', 'SH601989', 'SH601998', 'SH603993']
    env = AlphaEnv(csi100_2018, "2018-01-01", "2018-12-31", device)

    checkpoint_callback = CustomCallback(save_freq=10000, save_path='./logs/', name_prefix='maskable_ppo', verbose=1)

    model = MaskablePPO(
        'MlpPolicy',
        env,
        gamma=1.,
        ent_coef=0.02,
        verbose=1
    )
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
