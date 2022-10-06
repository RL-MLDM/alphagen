from itertools import count
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData


class AlphaPool:
    def __init__(self,
                 capacity: int,
                 stock_data: StockData,
                 target: Expression,
                 ic_lower_bound: float,
                 ic_min_increment: float
                 ):
        self.capacity = capacity
        self.size = 0

        self.data = stock_data
        self.target = self._normalize_by_day(target.evaluate(self.data))

        self.exprs: List[Optional[Expression]] = [None] * (capacity + 1)
        self.values: List[Optional[Tensor]] = [None] * (capacity + 1)
        self.ics_ret: np.ndarray = np.zeros(capacity + 1)
        self.ics_mut: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound
        self.ic_min_increment = ic_min_increment

    @property
    def device(self) -> torch.device:
        return self.data.device

    def try_new_expr(self, expr: Expression) -> float:
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value,
                                        ic_ret_threshold=self.ic_lower_bound,
                                        ic_mut_threshold=0.99
                                        )
        if ic_ret is None or ic_mut is None:
            return 0.

        self._add_factor(expr, value, ic_ret, ic_mut)

        self.optimize(alpha=5e-3, lr=5e-4, n_iter=500)
        self._pop()

        new_ic_ret = self.evaluate_ensemble()
        increment = new_ic_ret - self.best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        return increment

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            value = self._normalize_by_day(expr.evaluate(self.data))
            ic_ret, ic_mut = self._calc_ics(value,
                                            ic_ret_threshold=None,
                                            ic_mut_threshold=None
                                            )
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, value, ic_ret, ic_mut)
            assert self.size <= self.capacity
        self.optimize(alpha=5e-3, lr=5e-4, n_iter=100)

    def optimize(self, alpha, lr, n_iter) -> float:
        ics_ret = torch.from_numpy(self.ics_ret[:self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.ics_mut[:self.size, :self.size]).to(self.device)
        weights = torch.from_numpy(self.weights[:self.size]).to(self.device).requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7  # An arbitrary big value
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss_l1 = torch.norm(weights, p=1)
            loss = loss_ic + alpha * loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                break
            # if it % 100 == 0:
            #     print('>', loss_ic.item())

        self.weights[:self.size] = best_weights
        return best_weights[-1]

    def evaluate_ensemble(self) -> float:
        with torch.no_grad():
            ensemble_factor = self._normalize_by_day(sum(self.values[i] * self.weights[i] for i in range(self.size)))
            ensemble_ic = batch_pearsonr(ensemble_factor, self.target).mean().item()
            return ensemble_ic

    def _zero_eval(self) -> Tensor:
        return torch.zeros(self.data.n_days, self.data.n_stocks, 1, device=self.device)

    @staticmethod
    def _normalize_by_day(value: Tensor) -> Tensor:
        mean, std = masked_mean_std(value)
        value = (value - mean[:, None]) / std[:, None]
        nan_mask = torch.isnan(value)
        value[nan_mask] = 0.
        return value

    def _calc_ics(self,
                  value: Tensor,
                  ic_ret_threshold=None,
                  ic_mut_threshold=None
                  ) -> Tuple[Optional[float], Optional[List[float]]]:
        ic_ret = batch_pearsonr(value, self.target).mean().item()
        if ic_ret_threshold is not None and ic_ret < ic_ret_threshold:
            return None, None

        ic_mut = []
        for i in range(self.size):
            ic_mut_i = batch_pearsonr(value, self.values[i]).mean().item()
            if ic_mut_threshold is not None and ic_mut_i > ic_mut_threshold:
                return None, None
            ic_mut.append(ic_mut_i)

        return ic_ret, ic_mut

    def _add_factor(self,
                    expr: Expression,
                    value: Tensor,
                    ic_ret: float,
                    ic_mut: List[float],
                    ):
        n = self.size
        self.exprs[n] = expr
        self.values[n] = value
        self.ics_ret[n] = ic_ret
        for i in range(n):
            self.ics_mut[i][n] = self.ics_mut[n][i] = ic_mut[i]
        self.weights[n] = ic_ret  # An arbitrary init value
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = np.argmin(np.abs(self.weights))
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.values[i], self.values[j] = self.values[j], self.values[i]
        self.ics_ret[i], self.ics_ret[j] = self.ics_ret[j], self.ics_ret[i]
        self.ics_mut[:, [i, j]] = self.ics_mut[:, [j, i]]
        self.ics_mut[[i, j], :] = self.ics_mut[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]


if __name__ == '__main__':
    from alphagen.data.expression import *
    data = StockData(instrument='csi300', start_time='2009-01-01', end_time='2014-12-31')
    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    pool = AlphaPool(capacity=10,
                     stock_data=data,
                     target=target,
                     ic_lower_bound=0.,
                     ic_min_increment=0.)

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)
    volume = Feature(FeatureType.VOLUME)
    open_ = Feature(FeatureType.OPEN)
    pool.force_load_exprs([high, low, volume, open_, close])
    for i in range(10):
        increment = pool.try_new_expr(Div(Add(Less(Div(close, Ref(close, 30)), high),
                   Greater(Constant(5.0), Add(Add(Div(open_, Constant(-10.0)), Constant(-0.01)), Constant(0.01)))),
               Constant(-10.0)))
        print(increment, pool.best_ic_ret)
