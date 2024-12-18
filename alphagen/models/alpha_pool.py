from typing import Tuple, Dict, Any, Callable
from abc import ABCMeta, abstractmethod

import torch
from ..data.calculator import AlphaCalculator
from ..data.expression import Expression


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device = torch.device("cpu")
    ):
        self.size = 0
        self.capacity = capacity
        self.calculator = calculator
        self.device = device
        self.eval_cnt = 0
        self.best_ic_ret: float = -1.

    @property
    def vacancy(self) -> int:
        return self.capacity - self.size
    
    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        "Get a dictionary representing the state of this pool."

    @abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize this pool into a dictionary that can be dumped as json,
        i.e. no complex objects.
        """

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]: ...
