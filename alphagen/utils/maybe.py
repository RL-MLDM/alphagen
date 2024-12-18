from typing import Optional, TypeVar, Generic, Type, Callable, cast


_T = TypeVar("_T")
_TRes = TypeVar("_TRes")


class Maybe(Generic[_T]):
    def __init__(self, value: Optional[_T]) -> None:
        self._value = value

    @property
    def is_some(self) -> bool: return self._value is not None

    @property
    def is_none(self) -> bool: return self._value is None

    @property
    def value(self) -> Optional[_T]: return self._value

    def value_or(self, other: _T) -> _T:
        return cast(_T, self.value) if self.is_some else other

    def and_then(self, func: Callable[[_T], "Maybe[_TRes]"]) -> "Maybe[_TRes]":
        return func(cast(_T, self._value)) if self.is_some else Maybe(None)

    def map(self, func: Callable[[_T], _TRes]) -> "Maybe[_TRes]":
        return some(func(cast(_T, self._value))) if self.is_some else Maybe(None)

    def or_else(self, func: Callable[[], "Maybe[_T]"]) -> "Maybe[_T]":
        return self if self.is_some else func()


def some(value: _T) -> Maybe[_T]: return Maybe(value)
def none(_: Type[_T]) -> Maybe[_T]: return Maybe(None)
