import json
import os
from typing import Optional
from datetime import datetime

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, SingleAlphaPool
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
                 timestamp: Optional[str] = None,
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
        # if self.n_calls % self.show_freq == 0:
        #     self. show_pool_state()
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
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
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

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
        return self.training_env.envs[0].unwrapped.pool     # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000
):
    reseed_everything(seed)

    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    data = StockData(instrument=instruments,
                     start_time='2009-01-01',
                     end_time='2018-12-31')
    data_valid = StockData(instrument=instruments,
                           start_time='2019-01-01',
                           end_time='2019-12-31')
    data_test = StockData(instrument=instruments,
                          start_time='2020-01-01',
                          end_time='2021-12-31')

    if pool_capacity == 0:
        pool = SingleAlphaPool(
            capacity=pool_capacity,
            stock_data=data,
            target=target,
            ic_lower_bound=None
        )
    else:
        pool = AlphaPool(
            capacity=pool_capacity,
            stock_data=data,
            target=target,
            ic_lower_bound=None
        )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"ppo_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./logs/',
        valid_data=data_valid,
        valid_target=target,
        test_data=data_test,
        test_target=target,
        name_prefix=name_prefix,
        timestamp=timestamp,
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
        tensorboard_log=f'./tb_logs/ppo',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


if __name__ == '__main__':
    steps = {
        0: 250_000,
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for capacity in [0, 10, 20, 50, 100]:
        for seed in range(10):
            for instruments in ["csi300", "csi500"]:
                main(seed=seed, instruments=instruments, pool_capacity=capacity, steps=steps[capacity])
