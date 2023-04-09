from itertools import count
from typing import List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression
    ):
        self.capacity = capacity
        self.data = stock_data
        self.target = self._normalize_by_day(target.evaluate(self.data))

    @property
    def device(self) -> torch.device:
        return self.data.device

    @abstractmethod
    def to_dict(self) -> dict: ...

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...

    @abstractmethod
    def test_ensemble(self, data: StockData, target: Expression) -> Tuple[float, float]: ...

    @staticmethod
    def _normalize_by_day(value: Tensor) -> Tensor:
        mean, std = masked_mean_std(value)
        value = (value - mean[:, None]) / std[:, None]
        nan_mask = torch.isnan(value)
        value[nan_mask] = 0.
        return value


class AlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression,
        ic_lower_bound: Optional[float] = None
    ):
        super().__init__(capacity, stock_data, target)
        
        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.values: List[Optional[Tensor]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound

        self.eval_cnt = 0

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights[:self.size]),
            "best_ic_ret": self.best_ic_ret
        }

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size])
        }

    def try_new_expr(self, expr: Expression) -> float:
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None:
            return 0.

        self._add_factor(expr, value, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self._optimize(alpha=5e-3, lr=5e-4, n_iter=500)
            worst_idx = np.argmin(np.abs(new_weights))
            if worst_idx != self.capacity:
                self.weights[:self.size] = new_weights
                print(f"[Pool +] {expr}")
                if self.size > self.capacity:
                    print(f"[Pool -] {self.exprs[worst_idx]}")
            self._pop()

        new_ic_ret = self.evaluate_ensemble()
        increment = new_ic_ret - self.best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        return new_ic_ret

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            value = self._normalize_by_day(expr.evaluate(self.data))
            ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, value, ic_ret, ic_mut)
            assert self.size <= self.capacity
        self._optimize(alpha=5e-3, lr=5e-4, n_iter=500)

    def _optimize(self, alpha: float, lr: float, n_iter: int) -> np.ndarray:
        ics_ret = torch.from_numpy(self.single_ics[:self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.mutual_ics[:self.size, :self.size]).to(self.device)
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

            loss_l1 = torch.norm(weights, p=1)  # type: ignore
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

        return best_weights

    def test_ensemble(self, data: StockData, target: Expression) -> Tuple[float, float]:
        with torch.no_grad():
            factors: List[Tensor] = []
            for i in range(self.size):
                factor = self._normalize_by_day(self.exprs[i].evaluate(data))   # type: ignore
                weighted_factor = factor * self.weights[i]
                factors.append(weighted_factor)
            combined_factor: Tensor = sum(factors)  # type: ignore
            target_factor = target.evaluate(data)

            ic = batch_pearsonr(combined_factor, target_factor).mean().item()
            rank_ic = batch_spearmanr(combined_factor, target_factor).mean().item()
            return ic, rank_ic

    def evaluate_ensemble(self):
        with torch.no_grad():
            ensemble_factor = self._normalize_by_day(
                sum(self.values[i] * self.weights[i] for i in range(self.size)))    # type: ignore
            ensemble_ic = batch_pearsonr(ensemble_factor, self.target).mean().item()
            return ensemble_ic

    @staticmethod
    def _normalize_by_day(value: Tensor) -> Tensor:
        mean, std = masked_mean_std(value)
        value = (value - mean[:, None]) / std[:, None]
        nan_mask = torch.isnan(value)
        value[nan_mask] = 0.
        return value

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def _calc_ics(
        self,
        value: Tensor,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[List[float]]]:
        single_ic = batch_pearsonr(value, self.target).mean().item()
        thres = self.ic_lower_bound if self.ic_lower_bound is not None else 0.
        if not (self.size > 1 or self._under_thres_alpha) and abs(single_ic) < thres:
            return None, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = batch_pearsonr(value, self.values[i]).mean().item()  # type: ignore
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return None, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics

    def _add_factor(
        self,
        expr: Expression,
        value: Tensor,
        ic_ret: float,
        ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.values[n] = value
        self.single_ics[n] = ic_ret
        for i in range(n):
            self.mutual_ics[i][n] = self.mutual_ics[n][i] = ic_mut[i]
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
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]


class SingleAlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression,
        ic_lower_bound: Optional[float] = None,
        exclude_set: Optional[Set[Expression]] = None
    ):
        super().__init__(capacity, stock_data, target)

        self.cache = {}
        if exclude_set is None:
            self.exclude_set = []
        else:
            self.exclude_set = [self._normalize_by_day(expr.evaluate(self.data)) for expr in exclude_set]
        self.ic_lower_bound = ic_lower_bound

    def try_new_expr(self, expr: Expression) -> float:
        def calc_ic(x, y):
            return batch_pearsonr(x, y).mean().item()

        key = str(expr)
        if key in self.cache:
            return self.cache[key]
        value = self._normalize_by_day(expr.evaluate(self.data))
        for exc in self.exclude_set:
            if calc_ic(value, exc) > 0.9:
                self.cache[key] = -1.
                return -1.
        ic = calc_ic(value, self.target)
        self.cache[key] = ic
        return ic

    @property
    def size(self):
        return 1

    @property
    def weights(self):
        return np.array([1])

    @property
    def best_ic_ret(self):
        return max(self.cache.values())

    def to_dict(self) -> dict:
        return self.cache

    def test_ensemble(self, data: StockData, target: Expression) -> Tuple[float, float]:
        return 0., 0.


if __name__ == '__main__':
    from alphagen.data.expression import *
    data = StockData(instrument='csi300', start_time='2009-01-01', end_time='2014-12-31')
    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    pool = AlphaPool(capacity=10,
                     stock_data=data,
                     target=target,
                     ic_lower_bound=0.)

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
