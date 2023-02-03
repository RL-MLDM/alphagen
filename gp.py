import json
from collections import Counter
from pprint import pprint
import os

import numpy as np

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen.utils.random import reseed_everything
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

unary_ops = [Abs, Log]
binary_ops = [Add, Sub, Mul, Div, Greater, Less]
rolling_ops = [Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Delta, WMA, EMA]
rolling_binary_ops = [Cov, Corr]


def unary(cls):
    def _calc(a):
        n = len(a)
        return np.array([f'{cls.__name__}({a[i]})' for i in range(n)])

    return _calc


def binary(cls):
    def _calc(a, b):
        n = len(a)
        a = a.astype(str)
        b = b.astype(str)
        return np.array([f'{cls.__name__}({a[i]},{b[i]})' for i in range(n)])

    return _calc


def rolling(cls, day):
    def _calc(a):
        n = len(a)
        return np.array([f'{cls.__name__}({a[i]},{day})' for i in range(n)])

    return _calc


def rolling_binary(cls, day):
    def _calc(a, b):
        n = len(a)
        a = a.astype(str)
        b = b.astype(str)
        return np.array([f'{cls.__name__}({a[i]},{b[i]},{day})' for i in range(n)])

    return _calc


funcs = []
for op in unary_ops:
    funcs.append(make_function(function=unary(op), name=op.__name__, arity=1))
for op in binary_ops:
    funcs.append(make_function(function=binary(op), name=op.__name__, arity=2))
for op in rolling_ops:
    for day in [10, 20, 30, 40, 50]:
        funcs.append(make_function(function=rolling(op, day), name=op.__name__ + str(day), arity=1))
for op in rolling_binary_ops:
    for day in [10, 20, 30, 40, 50]:
        funcs.append(make_function(function=rolling_binary(op, day), name=op.__name__ + str(day), arity=2))

# with open('./logs/cache.json') as f:
#     cache = json.load(f)
instruments = "csi500"
seed = 9
reseed_everything(seed)

cache = {}
device = torch.device('cuda:0')
data = StockData(instruments, '2009-01-01', '2018-12-31', device=device)
data_valid = StockData(instruments, '2019-01-01', '2019-12-31', device=device)
data_test = StockData(instruments, '2020-01-01', '2021-12-31', device=device)

open_func = open

high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1

pool = AlphaPool(capacity=10,
                 stock_data=data,
                 target=target,
                 ic_lower_bound=None)


def _normalize_by_day(value: Tensor) -> Tensor:
    mean, std = masked_mean_std(value)
    value = (value - mean[:, None]) / std[:, None]
    nan_mask = torch.isnan(value)
    value[nan_mask] = 0.
    return value


target_factor = target.evaluate(data)
target_factor_valid = target.evaluate(data_valid)
target_factor_test = target.evaluate(data_test)


def _metric(x, y, w):
    key = y[0]
    # print(key)
    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.
    # print(key)
    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = _normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor).mean().item()
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic + token_len * 0.000


Metric = make_fitness(function=_metric, greater_is_better=True)


def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    v_valid = eval(top_key).evaluate(data_valid)
    v_test = eval(top_key).evaluate(data_test)
    ic_test = batch_pearsonr(v_test, target_factor_test).mean().item()
    ic_valid = batch_pearsonr(v_valid, target_factor_valid).mean().item()
    ric_test = batch_spearmanr(v_test, target_factor_test).mean().item()
    ric_valid = batch_spearmanr(v_valid, target_factor_valid).mean().item()
    return {"ic_test": ic_test, "ic_valid": ic_valid, "ric_test": ric_test, "ric_valid": ric_valid}


def try_pool(capacity):
    pool = AlphaPool(capacity=capacity,
                     stock_data=data,
                     target=target,
                     ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {"ic_test": ic_test, "ic_valid": ic_valid, "ric_test": ric_test, "ric_valid": ric_valid}


generation = 0

def ev():
    global generation
    generation += 1
    res = (
        [{"pool": 0, "res": try_single()}] +
        [{"pool": cap, "res": try_pool(cap)} for cap in (10, 20, 50, 100)]
    )
    print(res)
    dir = f"./logs/gp_{instruments}_{seed}"
    os.makedirs(dir, exist_ok=True)
    with open_func(f'{dir}/{generation}.json', 'w') as f:
        json.dump({"cache": cache, "res": res}, f)


if __name__ == '__main__':
    features = ['open', 'close', 'high', 'low', 'volume', 'vwap']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    all_tokens = features + constants

    X_train = np.array([all_tokens])
    y_train = np.array([[1]])

    est_gp = SymbolicRegressor(population_size=2000,
                               generations=60,
                               init_depth=(2, 6),
                               tournament_size=600,
                               stopping_criteria=1.,
                               p_crossover=0.3,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.01,
                               p_point_mutation=0.1,
                               p_point_replace=0.6,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.,
                               random_state=seed,
                               function_set=funcs,
                               metric=Metric,
                               const_range=None,
                               n_jobs=1)
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train))

    # exprs = []
    # for key in dict(Counter(cache).most_common(10)):
    #     exprs.append(eval(key))
    # pool.force_load_exprs(exprs)
    # pool.optimize(alpha=5e-3, lr=5e-4, n_iter=2000)
    # print(pool.to_dict())
    # with open_func('./logs/gp_pool_demo_p10_icsi300.json', 'w') as f:
    #     json.dump(pool.to_dict(), f)
