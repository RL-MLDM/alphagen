import os
from datetime import datetime

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 name_prefix: str = 'rl_model',
                 timestamp: str = None,
                 verbose: int = 0):
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
            self.show_pool_state()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (self.pool.weights[:self.pool.size].abs() > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPool:
        return self.training_env.envs[0].unwrapped.pool


if __name__ == '__main__':
    SEED = 0
    reseed_everything(SEED)

    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    data = StockData(instrument='csi300',
                     start_time='2009-01-01',
                     end_time='2018-12-31')
    pool = AlphaPool(capacity=20,
                     stock_data=data,
                     target=target,
                     ic_lower_bound=None,
                     ic_min_increment=None)
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    NAME_PREFIX = f'maskable_ppo_seed{SEED}'
    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./logs/',
        name_prefix=NAME_PREFIX,
        timestamp=TIMESTAMP,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=f'/DATA/xuehy/tb_logs/maskable_ppo',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
        tb_log_name=f'{NAME_PREFIX}_{TIMESTAMP}',
    )
