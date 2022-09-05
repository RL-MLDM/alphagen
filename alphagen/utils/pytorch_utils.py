from typing import Generic, TypeVar, Callable

from torch import nn


_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


class MapperModule(nn.Module, Generic[_TIn, _TOut]):
    def __init__(self, mapper: Callable[[_TIn], _TOut]) -> None:
        super().__init__()
        self.mapper = mapper

    def forward(self, input: _TIn) -> _TOut: return self.mapper(input)
