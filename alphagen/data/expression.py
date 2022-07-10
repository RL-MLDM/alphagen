from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

import torch
from torch import Tensor

from alphagen.data.stock_data import StockData, FeatureType


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["Expression", float]) -> "Sub":
        if isinstance(other, Expression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["Expression", float]) -> "Mul":
        if isinstance(other, Expression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["Expression", float]) -> "Div":
        if isinstance(other, Expression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): return not isinstance(self, (Constant, DeltaTime))


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str: return '$' + self._feature.name.lower()


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self._value, dtype=dtype, device=device)

    def __str__(self) -> str: return str(self._value)


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return str(self._delta_time)


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Expression) -> None:
        self._operand = operand

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"


class BinaryOperator(Operator):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        self._lhs = lhs
        self._rhs = rhs

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"


class RollingOperator(Operator):
    def __init__(self, operand: Expression, delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # F: feature count
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, F, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, F, S, W)
        return self._apply(values)                                  # (L, F, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"


class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs
        self._rhs = rhs
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: Expression,
                    data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # F: feature count
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, F, S)
        return values.unfold(0, self._delta_time, 1)                # (L, F, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)                                # (L, F, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, :, -1, None]
        left = (last < operand).count_nonzero()
        right = (last <= operand).count_nonzero()
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


Operators: List[Type[Expression]] = [
    # Unary
    Abs, Sign, Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov
]