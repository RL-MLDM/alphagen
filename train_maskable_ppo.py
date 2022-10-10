import json
import os
from datetime import datetime

import numpy as np
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
                 valid_data: StockData,
                 valid_target: Expression,
                 test_data: StockData,
                 test_target: Expression,
                 name_prefix: str = 'rl_model',
                 timestamp: str = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_data = valid_data
        self.valid_target = valid_target
        self.test_data = test_data
        self.test_target = test_target

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.show_freq == 0:
            self.show_pool_state()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        ic_valid, rank_ic_valid = self.pool.test_ensemble(self.valid_data, self.valid_target)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_data, self.test_target)
        self.logger.record('valid/ic', ic_valid)
        self.logger.record('valid/rank_ic', rank_ic_valid)
        self.logger.record('test/ic_', ic_test)
        self.logger.record('test/rank_ic_', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(pool.to_dict(), f)

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
    SEED = 1
    reseed_everything(SEED)

    INSTRUMENTS = 'csi500'

    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    data = StockData(instrument=INSTRUMENTS,
                     start_time='2009-01-01',
                     end_time='2018-12-31')
    data_valid = StockData(instrument=INSTRUMENTS,
                          start_time='2019-01-01',
                          end_time='2019-12-31')
    data_test = StockData(instrument=INSTRUMENTS,
                     start_time='2020-01-01',
                     end_time='2021-12-31')

    POOL_CAPACITY = 50
    pool = AlphaPool(capacity=POOL_CAPACITY,
                     stock_data=data,
                     target=target,
                     ic_lower_bound=None,
                     ic_min_increment=None)
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    MODEL_NAME = 'ppo_e_lstm'
    NAME_PREFIX = f'{MODEL_NAME}_s{SEED}_p{POOL_CAPACITY}_i{INSTRUMENTS}'
    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./logs/',
        valid_data=data_valid,
        valid_target=target,
        test_data=data_test,
        test_target=target,
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
        tensorboard_log=f'/DATA/xuehy/tb_logs/{MODEL_NAME}',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
        tb_log_name=f'{NAME_PREFIX}_{TIMESTAMP}',
    )
