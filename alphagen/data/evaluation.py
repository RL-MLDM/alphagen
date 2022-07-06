import pandas as pd
import torch
from qlib.data.dataset.loader import QlibDataLoader

from alphagen.data.expression import Expression
from alphagen.data.stock_data import StockData


def load_expr(expr: str, instrument: str, start_time: str, end_time: str) -> pd.DataFrame:
    return (QlibDataLoader(config={"feature": [expr]})      # type: ignore
            .load(instrument, start_time, end_time))


def correlation(joined: pd.DataFrame):
    return joined["factor"].corr(joined["target"], method="spearman")


class Evaluation:
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

        self.target = self._load('Ref($close,-20)/$close-1').iloc[:, 0].rename("target")

    def _load(self, expr: str) -> pd.DataFrame:
        return load_expr(expr, self.instrument, self.start_time, self.end_time)

    def evaluate(self, expr: Expression) -> float:
        factor = expr.evaluate(self._data)
        target = self._target.clone()
        nan_mask = factor.isnan() | target.isnan()
        factor[nan_mask] = torch.nan
        target[nan_mask] = torch.nan
        n = (~nan_mask).sum(dim=1)

        def rank_data(data: Tensor) -> Tensor:
            rank = data.argsort().argsort().float()         # [d, s]
            eq = data[:, None] == data[:, :, None]          # [d, s, s]
            eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
            rank = (eq @ rank[:, :, None]).squeeze(dim=2)
            rank[nan_mask] = 0
            return rank

        diff = rank_data(target) - rank_data(factor)
        coeff = 6 / (n * (n * n - 1))
        corrs = 1 - coeff * (diff * diff).sum(dim=1)
        return corrs.mean().item()


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)

    target = Ref(close, -20) / close - 1
    expr = Ref(abs(low), 10) + high / close

    ev = Evaluation('csi300', '2016-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))
