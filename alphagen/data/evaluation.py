import random
from typing import Tuple, List

import pandas as pd
import torch
from qlib.data.dataset.loader import QlibDataLoader

from alphagen.data.expression import Expression


def load_expr(expr: str, instrument: str, start_time: str, end_time: str) -> pd.DataFrame:
    return (QlibDataLoader(config={"feature": [expr]})      # type: ignore
            .load(instrument, start_time, end_time))


def correlation(joined: pd.DataFrame):
    return joined["factor"].corr(joined["target"], method="spearman")


class EvaluationBase:
    def evaluate(self, expr: Expression) -> float:
        raise NotImplementedError


class QLibEvaluation(EvaluationBase):
    instrument: str
    start_time: str
    end_time: str
    target: pd.Series

    def __init__(self,
                 instrument: str,
                 start_time: str, end_time: str,
                 target: Expression,
                 device: torch.device = torch.device("cpu")):
        self._data = StockData(instrument, start_time, end_time, device=device)
        self._target = target.evaluate(self._data)

        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time

    def _load(self, expr: str) -> pd.DataFrame:
        return load_expr(expr, self.instrument, self.start_time, self.end_time)

    def evaluate(self, expr: Expression) -> float:
        try:
            factor = expr.evaluate(self._data)
        except OutOfDataRangeError:
            return -1.
        target = self._target.clone()
        nan_mask = factor.isnan() | target.isnan()
        factor[nan_mask] = torch.nan
        target[nan_mask] = torch.nan
        n = (~nan_mask).sum(dim=1)

        def rank_data(data: torch.Tensor) -> torch.Tensor:
            rank = data.argsort().argsort().float()         # [d, s]
            eq = data[:, None] == data[:, :, None]          # [d, s, s]
            eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
            rank = (eq @ rank[:, :, None]).squeeze(dim=2)
            rank[nan_mask] = 0
            return rank                                     # [d, s]

        # Ignore the NaNs when calculating covariance/stddev
        def mean_std(rank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean = rank.sum(dim=1) / n
            std = ((((rank - mean[:, None]) * ~nan_mask) ** 2).sum(dim=1) / n).sqrt()
            return mean, std

        rx = rank_data(target)
        ry = rank_data(factor)
        rx_mean, rx_std = mean_std(rx)
        ry_mean, ry_std = mean_std(ry)
        cov = (rx * ry).sum(dim=1) / n - rx_mean * ry_mean
        stdmul = rx_std * ry_std
        stdmul[(rx_std < 1e-3) | (ry_std < 1e-3)] = 1

        corrs = cov / stdmul
        return corrs.mean().item()


class EvaluationPool(EvaluationBase):
    evaluations: List[EvaluationBase]

    def __init__(self, evaluations: List[EvaluationBase]):
        self.evaluations = evaluations

    def evaluate(self, expr: Expression) -> float:
        return random.choice(self.evaluations).evaluate(expr)


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)

    target = Ref(close, -20) / close - 1
    expr = Ref(abs(low), 10) + high / close

    ev1 = QLibEvaluation('csi300', '2016-01-01', '2018-12-31', target)
    ev2 = QLibEvaluation('csi300', '2014-01-01', '2015-12-31', target)
    ev_pool = EvaluationPool([ev1, ev2])
    for i in range(10):
        print(ev_pool.evaluate(expr))
