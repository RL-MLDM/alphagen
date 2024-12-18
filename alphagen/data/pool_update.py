from abc import ABCMeta, abstractmethod
from typing import List, Optional, cast
from dataclasses import dataclass, MISSING

from .expression import Expression


@dataclass
class PoolUpdate(metaclass=ABCMeta):
    @property
    @abstractmethod
    def old_pool(self) -> List[Expression]: ...

    @property
    @abstractmethod
    def new_pool(self) -> List[Expression]: ...

    @property
    @abstractmethod
    def old_pool_ic(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def new_pool_ic(self) -> float: ...

    @property
    def ic_increment(self) -> float:
        return self.new_pool_ic - (self.old_pool_ic or 0.)
    
    @abstractmethod
    def describe(self) -> str: ...
    
    def describe_verbose(self) -> str: return self.describe()

    def _describe_ic_diff(self) -> str:
        return (
            f"{self.old_pool_ic:.4f} -> {self.new_pool_ic:.4f} "
            f"(increment of {self.ic_increment:.4f})"
        )

    def _describe_pool(self, title: str, pool: List[Expression]) -> str:
        list_exprs = "\n".join([f"  {expr}" for expr in pool])
        return f"{title}\n{list_exprs}"


class _PoolUpdateStub:
    old_pool: List[Expression] = cast(List[Expression], MISSING)
    new_pool: List[Expression] = cast(List[Expression], MISSING)
    old_pool_ic: Optional[float] = cast(Optional[float], MISSING)
    new_pool_ic: float = cast(float, MISSING)


@dataclass
class SetPool(_PoolUpdateStub, PoolUpdate):
    old_pool: List[Expression]
    new_pool: List[Expression]
    old_pool_ic: Optional[float]
    new_pool_ic: float
    
    def describe(self) -> str:
        pool = self._describe_pool("Alpha pool:", self.new_pool)
        return f"{pool}\nIC of the combination: {self.new_pool_ic:.4f}"
    
    def describe_verbose(self) -> str:
        if len(self.old_pool) == 0:
            return self.describe()
        old_pool = self._describe_pool("Old alpha pool:", self.old_pool)
        new_pool = self._describe_pool("New alpha pool:", self.new_pool)
        perf = f"IC of the pools: {self._describe_ic_diff()})"
        return f"{old_pool}\n{new_pool}\n{perf}"


@dataclass
class AddRemoveAlphas(_PoolUpdateStub, PoolUpdate):
    added_exprs: List[Expression]
    removed_idx: List[int]
    old_pool: List[Expression]
    old_pool_ic: float
    new_pool_ic: float

    @property
    def new_pool(self) -> List[Expression]:
        remain = [True] * len(self.old_pool)
        for i in self.removed_idx:
            remain[i] = False
        return [expr for i, expr in enumerate(self.old_pool) if remain[i]] + self.added_exprs

    def describe(self) -> str:
        def describe_exprs(title: str, exprs: List[Expression]) -> str:
            if len(exprs) == 0:
                return ""
            if len(exprs) == 1:
                return f"{title}: {exprs[0]}\n"
            exprs_str = "\n".join([f"  {expr}" for expr in exprs])
            return f"{title}s:\n{exprs_str}\n"

        added = describe_exprs("Added alpha", self.added_exprs)
        removed = describe_exprs("Removed alpha", [self.old_pool[i] for i in self.removed_idx])
        perf = f"IC of the combination: {self._describe_ic_diff()}"
        return f"{added}{removed}{perf}"
    
    def describe_verbose(self) -> str:
        old = self._describe_pool("Old alpha pool:", self.old_pool)
        return f"{old}\n{self.describe()}"
