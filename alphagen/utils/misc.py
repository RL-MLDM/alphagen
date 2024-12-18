from typing import TypeVar, List, Iterable, Tuple, Callable, Optional
from types import FrameType
import inspect


_T = TypeVar("_T")


def reverse_enumerate(lst: List[_T]) -> Iterable[Tuple[int, _T]]:
    for i in range(len(lst) - 1, -1, -1):
        yield i, lst[i]


def find_last_if(lst: List[_T], predicate: Callable[[_T], bool]) -> int:
    for i in range(len(lst) - 1, -1, -1):
        if predicate(lst[i]):
            return i
    return -1


def get_arguments_as_dict(frame: Optional[FrameType] = None) -> dict:
    if frame is None:
        frame = inspect.currentframe().f_back   # type: ignore
    keys, _, _, values = inspect.getargvalues(frame)    # type: ignore
    res = {}
    for k in keys:
        if k != "self":
            res[k] = values[k]
    return res


def pprint_arguments(frame: Optional[FrameType] = None) -> dict:
    if frame is None:
        frame = inspect.currentframe().f_back   # type: ignore
    args = get_arguments_as_dict(frame)
    formatted_args = '\n'.join(f"    {k}: {v}" for k, v in args.items())
    print(f"[Parameters]\n{formatted_args}")
    return args
