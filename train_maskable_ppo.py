import os

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

from alphagen.data.expression import *
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.cache import LRUCache
from alphagen.utils.random import reseed_everything
from alphagen_qlib.evaluation import QLibEvaluation
from assets import ZZ300_2016


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 timestamp: str = None,
                 verbose: int = 0
                 ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_checkpoint()
        if self.n_calls % self.show_freq == 0:
            self.show_top_alphas()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('cache/size', len(self.cache))
        self.logger.record('cache/gt_0.05', self.cache.greater_than_count(0.05))
        self.logger.record('cache/top_1%', self.cache.quantile(0.99))
        self.logger.record('cache/top_100_avg', self.cache.top_k_average(100))

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)
        self.cache.save(path + '_cache.json')
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')

    def show_top_alphas(self):
        if self.verbose > 0:
            top_5 = self.cache.top_k(5)
            for key in top_5:
                print(key, top_5[key])

    @property
    def cache(self) -> LRUCache:
        return self.training_env.envs[0].unwrapped.eval.cache


if __name__ == '__main__':
    SEED = 0
    reseed_everything(SEED)

    device = torch.device('cuda:0')
    csi300_2016 = ZZ300_2016
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    ev = QLibEvaluation(
        instrument=csi300_2016,
        start_time='2016-01-01',
        end_time='2018-12-31',
        target=target,
        device=device,
    )
    ev.cache.preload('/DATA/xuehy/preload/zz300_static_20160101_20181231.json')
    env = AlphaEnv(ev)

    NAME_PREFIX = f'maskable_ppo_seed{SEED}'
    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./logs/',
        name_prefix=NAME_PREFIX,
        timestamp=TIMESTAMP,
        verbose=1
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        gamma=1.,
        ent_coef=0.02,
        batch_size=128,
        tensorboard_log=f'/DATA/xuehy/tb_logs/maskable_ppo',
        device=device,
        verbose=1
    )
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
        tb_log_name=f'{NAME_PREFIX}_{TIMESTAMP}'
    )
