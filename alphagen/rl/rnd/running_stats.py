import math
from typing import Union
import numpy as np


class RunningStats:
    count: float
    max: float
    min: float
    mean: float
    moment2: float

    def __init__(self) -> None:
        self.clear()

    def add_entry(self, entry: Union[float, np.ndarray]) -> None:
        if isinstance(entry, (int, float)):
            entry = np.array((entry,), dtype=float)
        else:
            entry = entry.flatten()

        self.count += entry.size
        self.max = max(entry.max(), self.max)
        self.min = min(entry.min(), self.min)
        self.mean += (entry.mean() - self.mean) * entry.size / self.count
        self.moment2 += ((entry ** 2).mean() - self.moment2) * entry.size / self.count

    def decay_count(self, factor: float) -> None:
        self.count *= factor

    def clear(self) -> None:
        self.count = 0.
        self.max = -float("inf")
        self.min = float("inf")
        self.mean = 0.
        self.moment2 = 0.

    @property
    def stddev(self) -> float: return math.sqrt(self.moment2 - self.mean ** 2)

    @property
    def sum(self) -> float: return self.mean * self.count
