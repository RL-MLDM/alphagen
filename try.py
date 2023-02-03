import json
from pathlib import Path
from pprint import pprint

from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std

close = Feature(FeatureType.CLOSE)
open_ = Feature(FeatureType.OPEN)
high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
vwap = Feature(FeatureType.VWAP)
returns = Ref(close, -20) / close - 1

device = torch.device('cuda:0')
sd = StockData(instrument='csi300',
               start_time='2020-01-01',
               end_time='2021-12-31', device=device)


def into_weights_and_exprs(j):
    weights = []
    exprs = []
    for i in range(len(j["weights"])):
        weights.append(j["weights"][i])
        expr = j["exprs"][i].replace("$", "").replace("open", "open_")
        exprs.append(eval(expr))
    return weights, exprs


def normalized_eval(expr: Expression) -> Tensor:
    value = expr.evaluate(sd)           # [days, stocks]
    mean, std = masked_mean_std(value)
    value = (value - mean[:, None]) / std[:, None]
    nan_mask = torch.isnan(value)
    value[nan_mask] = 0.
    return value


def eval_combination(weights: List[float], exprs: List[Expression]):
    factors = []
    for i, expr in enumerate(exprs):
        factor = normalized_eval(expr)
        weighted_factor = factor * weights[i]
        factors.append(weighted_factor)
    combined_factor = sum(factors)
    target_factor = returns.evaluate(sd)
    ic = batch_pearsonr(combined_factor, target_factor).mean().item()
    ric = batch_spearmanr(combined_factor, target_factor).mean().item()
    return combined_factor, ic, ric


if __name__ == '__main__':
    alphas_dict = {}
    for file in Path("./logs").rglob("2000896_steps_pool.json"):
        stem = file.parent.stem
        # print(stem)
        splitted = stem.split("_")[3:6]
        seed = int(splitted[0][1:])
        pool = int(splitted[1][1:])
        csi = int(splitted[2][4:])
        if pool == 0 or csi != 300:
            continue

        with open(file, "r") as f:
            alphas = json.load(f)
            alphas_dict[(csi, pool, seed)] = alphas

    combinations = {}

    for k, v in alphas_dict.items():
        w, e = into_weights_and_exprs(v)
        comb, ic, rank_ic = eval_combination(w, e)
        combinations[k] = (comb, ic, rank_ic)
    pprint({key: combinations[key][1] for key in combinations})

    for csi in [300]:
        for pool in [10, 20, 50, 100]:
            ic = 0.
            rank_ic = 0.
            count = 0
            for seed in range(5):
                tup = (csi, pool, seed)
                if tup == (300, 20, 1):
                    continue
                if tup not in combinations:
                    assert False
                count += 1
                _, ic_, rank_ic_ = combinations.get((csi, pool, seed), (None, 0., 0.))
                ic += ic_
                rank_ic += rank_ic_
            # if pool == 20:
            #     assert count == 4
            # else:
            #     assert count == 5
            ic /= count
            rank_ic /= count
            print(f"CSI{csi}, Pool {pool}, Avg IC = {ic}, Avg Rank IC = {rank_ic}")