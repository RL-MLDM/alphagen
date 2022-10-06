from typing import Optional

import numpy as np

from alphagen.data.evaluation import Evaluation
from alphagen.data.expression import *
from alphagen.utils.cache import LRUCache, LRUCACHE_NOT_FOUND
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


class QLibEvaluation(Evaluation):
    instrument: str
    start_time: str
    end_time: str
    target: torch.Tensor

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str, end_time: str,
                 target: Expression,
                 print_expr: bool,
                 device: torch.device):
        self.data = StockData(instrument, start_time, end_time, device=device)
        self.target = target.evaluate(self.data)

        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time

        self.cache = LRUCache(100000)
        self.print_expr = print_expr

    def evaluate(self, expr: Expression, print_expr_override: Optional[bool] = None) -> float:
        key = str(expr)
        found = self.cache.get(key)
        if found != LRUCACHE_NOT_FOUND:
            return found
        try:
            factor = expr.evaluate(self.data)
            corrs = batch_pearsonr(factor, self.target)
            ic = corrs.mean().item()

            if np.isnan(ic):
                raise ValueError(f'Nan factor: {key}')

            print_expr = self.print_expr if print_expr_override is None else print_expr_override
            if print_expr:
                print(key, ic)

            self.cache.put(key, ic)
        except OutOfDataRangeError:
            ic = -1.
        except ValueError as e:
            ic = -1.

        return ic


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)
    volume = Feature(FeatureType.VOLUME)

    target = Ref(close, -20) / close - 1
    expr = Ref(Greater(Greater(Constant(-30.0), high), Constant(-5.0)), 40)

    ev = QLibEvaluation(
        instrument='csi300',
        start_time='2019-01-01',
        end_time='2021-12-31',
        target=target,
        print_expr=True,
        device=torch.device('cuda:0')
    )
    print(ev.evaluate(expr))
