from typing import Union, List

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

    def evaluate(self, expr: Expression) -> float:
        key = str(expr)
        existing_val = self.cache.get(key)
        if existing_val != LRUCACHE_NOT_FOUND:
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

    ev = QLibEvaluation('csi100', '2018-01-01', '2018-12-31', target)
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
    ev = QLibEvaluation(csi100_2018, '2018-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))
