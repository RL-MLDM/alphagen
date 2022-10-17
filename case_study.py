import json

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.stock_data import StockData
from gp import _normalize_by_day

open_func = open

high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1

PATH = './logs/ppo_e_lstm_s2_p10_icsi300_20221011111915/2000896_steps_pool.json'

data_train = StockData(instrument='csi300',
                       start_time='2009-01-01',
                       end_time='2018-12-31')
data_test = StockData(instrument='csi300',
                      start_time='2020-01-01',
                      end_time='2021-12-31')


def try_pool(exprs):
    n = len(exprs)
    a_pool = AlphaPool(capacity=n,
                       stock_data=data_train,
                       target=target,
                       ic_lower_bound=None,
                       ic_min_increment=None)
    a_pool.force_load_exprs(exprs)
    a_pool.optimize(alpha=5e-3, lr=5e-4, n_iter=2000)
    return a_pool.test_ensemble(data_test, target)


if __name__ == '__main__':
    data_test = StockData(instrument='csi300',
                          start_time='2020-01-01',
                          end_time='2021-12-31')

    with open_func(PATH) as f:
        pool = json.load(f)

    keys = pool['exprs']
    exprs = [eval(key.replace('$', '')) for key in keys]
    weights = pool['weights']

    target_factor = target.evaluate(data_test)
    factors = []
    for i, expr in enumerate(exprs):
        factor = expr.evaluate(data_test)
        factor = _normalize_by_day(factor)

        ic = batch_pearsonr(factor, target_factor).mean().item()
        print(i, ic, try_pool([exprs[j] for j in range(10) if j != i])[0], weights[i])

        weighted_factor = factor * weights[i]
        factors.append(weighted_factor)
    combined_factor = sum(factors)

    ic = batch_pearsonr(combined_factor, target_factor).mean().item()
    rank_ic = batch_spearmanr(combined_factor, target_factor).mean().item()
    print(ic, rank_ic)
