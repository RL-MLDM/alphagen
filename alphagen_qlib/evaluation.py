from typing import Union, List

import numpy as np
import pandas as pd
import torch

from alphagen.data.evaluation import Evaluation
from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.utils.cache import LRUCache, LRUCACHE_NOT_FOUND
from alphagen.utils.correlation import batch_spearman
from alphagen_qlib.stock_data import StockData


class QLibEvaluation(Evaluation):
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
        self.visited_1 = np.zeros_like(self._target.cpu())
        self.sum_1 = 0
        self.visited_0 = np.zeros_like(self._target.cpu())
        self.sum_0 = 0

    def evaluate(self,
                 expr: Expression,
                 use_curiosity=True,
                 print_expr: bool = True,
                 add_noise: bool = False,
                 ) -> float:
        key = str(expr)
        try:
            factor = expr.evaluate(self._data)
            if add_noise:
                factor += 1e-6 * torch.randn_like(factor)
            corrs = batch_spearman(factor, self._target)
            ic = corrs.mean().item()

            if np.isnan(ic):
                raise ValueError(f'Nan factor: {key}')
            factor = factor.cpu().numpy()
            median = np.nanmedian(factor, axis=1, keepdims=True)
            mask_1, mask_0 = np.nan_to_num(factor >= median), np.nan_to_num(factor < median)

            # if ic >= 0.:
            weighted_count = (np.sum(self.visited_1[mask_1]) + np.sum(self.visited_0[mask_0]))
            # else:
            #     weighted_count = (np.sum(self.visited_1[mask_0]) + np.sum(self.visited_0[mask_1]))
            mean_count = (self.sum_1 + self.sum_0) / 2
            curiosity = 0.5 * (1e-4 + mean_count) / (1e-4 + weighted_count)
            if print_expr:
                print(key, ic, curiosity)

            self.visited_1 += mask_1
            self.sum_1 += mask_1.sum()
            self.visited_0 += mask_0
            self.sum_0 += mask_0.sum()

            ic = abs(ic)
            self.cache.put(key, ic)
        except OutOfDataRangeError:
            ic, curiosity = -1., 0.
        except ValueError as e:
            ic, curiosity = -1., 0.

        if use_curiosity:
            return ic + curiosity
        else:
            return ic


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)
    volume = Feature(FeatureType.VOLUME)

    target = Ref(close, -20) / close - 1
    expr = Ref(Greater(Greater(Constant(-30.0), high), Constant(-5.0)), 40)

    ev = QLibEvaluation('csi300', '2019-01-01', '2021-12-31', target)
    print(ev.evaluate(expr, use_curiosity=False))
