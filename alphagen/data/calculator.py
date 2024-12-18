from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence
from torch import Tensor
import torch

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        return self.calc_single_IC_ret(expr), self.calc_single_rIC_ret(expr)

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'


class TensorAlphaCalculator(AlphaCalculator):
    def __init__(self, target: Optional[Tensor]) -> None:
        self._target = target

    @property
    @abstractmethod
    def n_days(self) -> int: ...

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("A target must be set before calculating non-mutual IC.")
        return self._target

    @abstractmethod
    def evaluate_alpha(self, expr: Expression) -> Tensor:
        'Evaluate an alpha into a `Tensor` of shape (days, stocks).'

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()
    
    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()
    
    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item()
    
    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))
    
    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)
    
    def calc_single_IC_ret_daily(self, expr: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._calc_rIC(self.evaluate_alpha(expr), self.target)
    
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_mutual_IC_daily(self, expr1: Expression, expr2: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)
        
    def calc_pool_all_ret_with_ir(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float, float, float]:
        "Returns IC, ICIR, Rank IC, Rank ICIR"
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std
