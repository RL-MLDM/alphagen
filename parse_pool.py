import json
import os
from collections import Counter, defaultdict

from matplotlib import pyplot as plt

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.stock_data import StockData
from playground import _normalize_by_day

open_func = open

high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1


def search_model(path, prefix):
    filenames = os.listdir(path)
    found = sorted([t for t in filenames if '_'.join(t.split('_')[:-1]) == prefix])
    if len(found) == 1:
        return found[0]
    else:
        for t in found:
            print(t)
        raise ValueError


MODEL_NAME = 'ppo_s_lstm'
LOG_DIR = f'/DATA/xuehy/logs'


if __name__ == '__main__':
    reseed_everything(0)

    data_train = StockData(instrument='csi300',
                           start_time='2009-01-01',
                           end_time='2018-12-31')
    data_test = StockData(instrument='csi300',
                          start_time='2020-01-01',
                          end_time='2021-12-31')

    ic_res = defaultdict(float)
    ric_res = defaultdict(float)

    for seed in range(5):
        MODEL_DIR = search_model(LOG_DIR, f'{MODEL_NAME}_s{seed}_p0_icsi300')
        PATH = f'{LOG_DIR}/{MODEL_DIR}'
        print(MODEL_DIR)
        for step in (2000896, ):
            path = f'{PATH}/{step}_steps_pool.json'
            with open_func(path) as f:
                cache = json.load(f)

            top_key = Counter(cache).most_common(1)[0][0]
            top1_ic = batch_pearsonr(eval(top_key.replace('$', '')).evaluate(data_test), target.evaluate(data_test)).mean().item()
            top1_rank_ic = batch_spearmanr(eval(top_key.replace('$', '')).evaluate(data_test), target.evaluate(data_test)).mean().item()
            print(f'top1 step: {step}, ic: {top1_ic}, rank_ic: {top1_rank_ic}')

            ic_res[(step, 0)] += top1_ic
            ric_res[(step, 0)] += top1_rank_ic

            for capacity in (10, ):
                keys = dict(Counter(cache).most_common(capacity)).keys()
                a_pool = AlphaPool(capacity=capacity,
                                   stock_data=data_train,
                                   target=target,
                                   ic_lower_bound=None,
                                   ic_min_increment=None)
                a_pool.force_load_exprs([eval(key.replace('$', '')) for key in keys])
                a_pool.optimize(alpha=5e-3, lr=5e-4, n_iter=2000)
                ic_test, rank_ic_test = a_pool.test_ensemble(data_test, target)

                ic_res[(step, capacity)] += ic_test
                ric_res[(step, capacity)] += rank_ic_test

                print(f'p{capacity} step: {step}, ic: {ic_test}, rank_ic: {rank_ic_test}')
                print(a_pool.to_dict())

    for step in (2000896, 1001472):
        for sz in (0, 10, 20, 50, 100):
            print(sz, step, ic_res[(step, sz)] / 5, ric_res[(step, sz)] / 5)
    # plt.plot(steps, ics_test, label='ic')
    # plt.plot(steps, rics_test, label='rank ic')
    #
    # plt.legend()
    # plt.show()
