from collections import OrderedDict
from typing import Tuple, Union, List

import pandas as pd
import torch
from torch import Tensor

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.data.stock_data import StockData


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.best_key = ''
        self.best_value = -2.

    def __len__(self):
        return len(self.cache)

    def get(self, key: str) -> int:
        if key not in self.cache:
            return -2
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        if value > self.best_value:
            self.best_key = key
            self.best_value = value


def batch_spearman(x: Tensor, y: Tensor) -> Tensor:
    x = x.clone()
    y = y.clone()
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = torch.nan
    y[nan_mask] = torch.nan
    n = (~nan_mask).sum(dim=1)

    def rank_data(data: Tensor) -> Tensor:
        rank = data.argsort().argsort().float()         # [d, s]
        eq = data[:, None] == data[:, :, None]          # [d, s, s]
        eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
        rank = (eq @ rank[:, :, None]).squeeze(dim=2)
        rank[nan_mask] = 0
        return rank                                     # [d, s]

    # Ignore the NaNs when calculating covariance/stddev
    def mean_std(rank: Tensor) -> Tuple[Tensor, Tensor]:
        mean = rank.sum(dim=1) / n
        std = ((((rank - mean[:, None]) * ~nan_mask) ** 2).sum(dim=1) / n).sqrt()
        return mean, std

    rx = rank_data(y)
    ry = rank_data(x)
    rx_mean, rx_std = mean_std(rx)
    ry_mean, ry_std = mean_std(ry)
    cov = (rx * ry).sum(dim=1) / n - rx_mean * ry_mean
    stdmul = rx_std * ry_std
    stdmul[(rx_std < 1e-3) | (ry_std < 1e-3)] = 1

    corrs = cov / stdmul
    return corrs


class Evaluation:
    instrument: str
    start_time: str
    end_time: str
    target: pd.Series

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str, end_time: str,
                 target: Expression,
                 device: torch.device = torch.device("cpu")):
        self._data = StockData(instrument, start_time, end_time, device=device)
        self._target = target.evaluate(self._data)

        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time

        self.cache = LRUCache(100000)

    def evaluate(self, expr: Expression) -> float:
        key = str(expr)
        existing_val = self.cache.get(key)
        if existing_val != -2:
            return existing_val

        try:
            factor = expr.evaluate(self._data)
            target = self._target.clone()
            corrs = batch_spearman(factor, target)
            ret = corrs.mean().item()
        except OutOfDataRangeError:
            ret = -1.

        self.cache.put(key, ret)
        return ret


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)
    volume = Feature(FeatureType.VOLUME)

    target = Ref(close, -20) / close - 1
    expr = Greater(Sub(Sign(Med(volume, 30)), Constant(-0.5)), Constant(-2.0))

    ev = Evaluation('csi100', '2018-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))

    csi100_2018 = ['SZ000001', 'SZ000002', 'SZ000063', 'SZ000069', 'SZ000166', 'SZ000333', 'SZ000538', 'SZ000625', 'SZ000651',
     'SZ000725', 'SZ000776', 'SZ000858', 'SZ000895', 'SZ001979', 'SZ002024', 'SZ002027', 'SZ002142', 'SZ002252',
     'SZ002304', 'SZ002352', 'SZ002415', 'SZ002558', 'SZ002594', 'SZ002736', 'SZ002739', 'SZ002797', 'SZ300059',
     'SH600000', 'SH600010', 'SH600011', 'SH600015', 'SH600016', 'SH600018', 'SH600019', 'SH600023', 'SH600028',
     'SH600030', 'SH600036', 'SH600048', 'SH600050', 'SH600061', 'SH600104', 'SH600115', 'SH600276', 'SH600309',
     'SH600340', 'SH600518', 'SH600519', 'SH600585', 'SH600606', 'SH600637', 'SH600663', 'SH600690', 'SH600703',
     'SH600795', 'SH600837', 'SH600887', 'SH600893', 'SH600900', 'SH600919', 'SH600958', 'SH600999', 'SH601006',
     'SH601009', 'SH601018', 'SH601088', 'SH601111', 'SH601166', 'SH601169', 'SH601186', 'SH601211', 'SH601225',
     'SH601229', 'SH601288', 'SH601318', 'SH601328', 'SH601336', 'SH601390', 'SH601398', 'SH601601', 'SH601618',
     'SH601628', 'SH601633', 'SH601668', 'SH601669', 'SH601688', 'SH601727', 'SH601766', 'SH601788', 'SH601800',
     'SH601818', 'SH601857', 'SH601881', 'SH601899', 'SH601901', 'SH601985', 'SH601988', 'SH601989', 'SH601998',
     'SH603993']
    ev = Evaluation(csi100_2018, '2018-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))
