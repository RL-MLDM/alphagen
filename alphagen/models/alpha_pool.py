import math
from typing import List, Optional
from copy import deepcopy

import torch
from torch import Tensor, nn, optim

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.data.stock_data import StockData
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen.utils.correlation import batch_pearsonr


class AlphaPool:
    def __init__(
        self,
        stock_data: StockData,
        target: Expression,
        record_path: Optional[str] = None,
    ) -> None:
        self._data = stock_data
        self._record_path = record_path
        self._target_expr = target
        self._target = self._normalized_eval(target)
        self._exprs: List[Expression] = []
        self._evals: Tensor = self._zero_eval()
        self._model: _LinearModel = _LinearModel(self.device)
        self._current_ic: float = 0.

    @property
    def device(self) -> torch.device: return self._data.device

    def _zero_eval(self) -> Tensor:
        return torch.zeros(self._data.n_days, self._data.n_stocks, 1, device=self.device)

    def _normalized_eval(self, expr: Expression) -> Tensor:
        value = expr.evaluate(self._data)   # [days, stocks]
        mean, std = masked_mean_std(value)
        value = (value - mean[:, None]) / std[:, None]
        nan_mask = torch.isnan(value)
        value[nan_mask] = 0.                # TODO: Is this mask really necessary?
        return value

    def _grow(self) -> None:
        self._evals = torch.cat((self._evals, self._zero_eval()), dim=2)
        self._model.extend_one()

    def _record_expr(self, expr: Expression, score: float) -> None:
        if self._record_path is None:
            return
        with open(self._record_path, "a") as f:
            f.write(f"Expr: {score:.6f}\t{expr}\n")

    def _record_pool(self) -> None:
        if self._record_path is None:
            return
        ics = [batch_pearsonr(self._evals[:, :, i], self._target).mean().item()
               for i in range(self._evals.shape[2])]
        weights: List[float] = self._model._weights.cpu().tolist()
        weights.append(1. - sum(weights))
        with open(self._record_path, "a") as f:
            f.write(f"Pool: Linear model IC = {self._current_ic:.6f}\n")
            for i, e, w in zip(ics, self._exprs, weights):
                f.write(f"    Singular IC = {i:.6f}, Weight = {w:.6f}, Expression: {e}\n")

    def try_new_expr(
        self,
        expr: Expression,
        min_ic_first: float = 0.03,
        min_ic_increment: float = 0.005
    ) -> float:
        try:
            norm_eval = self._normalized_eval(expr)
            self._evals[:, :, -1] = norm_eval
        except OutOfDataRangeError:
            return -1.

        if self._model._dim == 1:   # Empty pool for now
            ic = batch_pearsonr(norm_eval, self._target).mean().item()
            if ic < min_ic_first:
                self._record_expr(expr, ic)
                return ic
        else:
            model = deepcopy(self._model)
            sgd = optim.SGD(model.parameters(), lr=1.)
            prev_highest, ic = 0., self._current_ic
            count = 0
            while True:
                combined = model(self._evals)
                ic = batch_pearsonr(combined, self._target).mean()
                (-ic).backward()
                sgd.step()
                sgd.zero_grad()
                ic = ic.item()
                if count > 100:
                    break
                if ic > prev_highest:
                    if ic > prev_highest + 1e-6:
                        count = 0
                    prev_highest = ic
                count += 1
            diff = ic - self._current_ic
            if diff < min_ic_increment:
                self._record_expr(expr, ic)
                return diff
            self._model = model

        self._exprs.append(expr)
        self._grow()
        result = ic - self._current_ic
        self._current_ic = ic
        self._record_pool()
        return result


class _LinearModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._dim: int = 1
        self._weights = nn.Parameter(torch.zeros(0, device=device))

    def forward(self, x: Tensor) -> Tensor:
        if self._dim < 2:
            return x
        last = 1. - self._weights.sum()
        all_weights = torch.cat((self._weights, last[None]))
        return x @ all_weights

    def extend_one(self) -> None:
        with torch.no_grad():
            last = 1. - self._weights.sum()
            new_weights = torch.cat((self._weights, last[None]))
        self._weights = nn.Parameter(new_weights)
        self._dim += 1
