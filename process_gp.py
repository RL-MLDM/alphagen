from typing import Tuple, Dict
import torch
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.random import reseed_everything
import os
from glob import glob
import json
from contextlib import redirect_stdout
from tqdm import tqdm
from collections import Counter

devnull = open(os.devnull, 'w')
device = torch.device('cuda:0')

data: Dict[str, Tuple[StockData, StockData, StockData]] = {}


def get_data(instruments: str) -> Tuple[StockData, StockData, StockData]:
    if instruments not in data:
        data[instruments] = (
            StockData(instruments, '2009-01-01', '2018-12-31', device=device),
            StockData(instruments, '2019-01-01', '2019-12-31', device=device),
            StockData(instruments, '2020-01-01', '2021-12-31', device=device)
        )
    return data[instruments]


target = Ref(Feature(FeatureType.CLOSE), -20) / Feature(FeatureType.CLOSE) - 1


def parse_expr(expr: str) -> Expression:
    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    volume = Feature(FeatureType.VOLUME)
    open = Feature(FeatureType.OPEN)
    close = Feature(FeatureType.CLOSE)
    vwap = Feature(FeatureType.VWAP)
    return eval(expr.replace("$", ""))


def main(
    in_path: str,
    out_path: str,
    instruments: str = "csi300",
    capacity: int = 10,
    seed: int = 0,
    use_filtering: bool = False
):
    print(f"[Running] {instruments}, pool capacity: {capacity}, seed: {seed}")
    reseed_everything(seed)

    train, valid, test = get_data(instruments)
    pool = AlphaPool(capacity, train, target)

    results = {}

    def evaluate(step: int):
        train_ic = pool.evaluate_ensemble()
        test_ic, test_ric = pool.test_ensemble(test, target)
        val_ic, val_ric = pool.test_ensemble(valid, target)
        print(f"[{step}] Train IC: {train_ic:.4f}, Val: ({val_ic:.4f}, {val_ric:.4f}) Test: ({test_ic:.4f}, {test_ric:.4f})")
        results[step] = {
            "train_ic": train_ic,
            "valid_ic": val_ic,
            "valid_ric": val_ric,
            "test_ic": test_ic,
            "test_ric": test_ric,
            "pool": pool.to_dict()
        }

    with open(in_path, "r") as f:
        cache = json.load(f)
    cache = {k: v for k, v in cache.items() if v != 0.0 and v != -1.0}

    alphas = []
    for e, ic in Counter(cache).most_common():
        expr = parse_expr(e)
        values = AlphaPool._normalize_by_day(expr.evaluate(train))
        if use_filtering:
            mutual = 0.
            for other in alphas:
                mutual = batch_pearsonr(values, other[1]).mean().item()
                if mutual > 0.7:
                    break
            if mutual > 0.7:
                continue
        alphas.append((expr, values))
        if len(alphas) >= capacity:
            break
    print(f"Actual pool size: {len(alphas)}")
    pool.force_load_exprs([e[0] for e in alphas])
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)
    evaluate(len(cache))

    with open(out_path, "w") as f:
        json.dump(results, f)

    best = max(results.items(), key=lambda x: x[1]["valid_ic"])
    print(f"Best (step={best[0]}): IC = {best[1]['test_ic']:.4f}, Rank IC = {best[1]['test_ric']:.4f}")


if __name__ == "__main__":
    train, valid, test = get_data("csi300")
    dct = ([3, 1, 5, 6, 9, 2, 8, 7, 0, 4],
            [157696, 215040, 174080, 98304, 243712,
            344064, 126976, 155648, 628736, 120832])
    for seed, step in zip(*dct):
        dir = glob(f"logs/ppo_csi300_100_{seed}*")[0]
        file = dir + f"/{step}_steps_pool.json"
        with open(file, "r") as f:
            js = json.load(f)
        # js = list(js.values())[0]["pool"]
        values = None
        for e, w in zip(js["exprs"], js["weights"]):
            expr = parse_expr(e)
            v = w * AlphaPool._normalize_by_day(expr.evaluate(test))
            values = v if values is None else values + v
        assert values is not None
        test.make_dataframe([values], ["score"])["score"].to_pickle(f"{seed}.pkl")
    # for instruments in ["csi300", "csi500"]:
    #     for seed in range(0, 10):
    #         in_path = glob(f"logs/ppo_{instruments}_0_{seed}_*")[0]
    #         in_path += "/251904_steps_pool.json"
    #         for cap in [1]:
    #             out_path = f"logs/ppo_{instruments}_baseline/filter_p{cap}_{seed}.json"
    #             main(in_path, out_path, instruments, cap, seed, use_filtering=False)
