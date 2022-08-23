from collections import OrderedDict, Counter

import numpy as np
import json


LRUCACHE_NOT_FOUND = -2
NAN = float('nan')


class LRUCache:
    def __init__(self, capacity: int):
        self.cache_valid = OrderedDict()
        self.cache_nan = OrderedDict()
        self.capacity = capacity
        self.preload_valid = {}
        self.preload_nan = {}

    def __len__(self):
        return len(self.cache_valid)

    def top_k(self, k: int) -> dict:
        return dict(Counter(self.cache_valid).most_common(k))

    def top_k_average(self, k: int) -> float:
        top_k_values = self.top_k(k).values()
        return sum(top_k_values) / len(top_k_values)

    def greater_than(self, threshold: float) -> dict:
        return {k: self.cache_valid[k] for k in self.cache_valid if self.cache_valid[k] >= threshold}

    def greater_than_count(self, threshold: float) -> int:
        return sum(self.cache_valid[k] >= threshold for k in self.cache_valid)

    def quantile(self, q: float):
        return np.quantile(list(self.cache_valid.values()), q)

    def get(self, key: str) -> float:
        if key in self.cache_valid:
            self.cache_valid.move_to_end(key)
            return self.cache_valid[key]
        elif key in self.cache_nan:
            self.cache_nan.move_to_end(key)
            return NAN
        elif key in self.preload_valid:
            value = self.preload_valid[key]
            self._put_valid(key, value)
            return value
        elif key in self.preload_nan:
            self._put_nan(key)
            return NAN
        else:
            return LRUCACHE_NOT_FOUND

    def put(self, key: str, value: int) -> None:
        if value != value:  # NaN
            self._put_nan(key)
        else:
            self._put_valid(key, value)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'valid': self.cache_valid, 'nan': self.cache_nan}, f, ensure_ascii=False, indent=False)

    def preload(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            preload = json.load(f)
            self.preload_valid = preload['valid']
            self.preload_nan = preload['nan']

    def _put_valid(self, key: str, value: int) -> None:
        self.cache_valid[key] = value
        self.cache_valid.move_to_end(key)
        if len(self.cache_valid) > self.capacity:
            self.cache_valid.popitem(last=False)

    def _put_nan(self, key: str) -> None:
        self.cache_nan[key] = NAN
        self.cache_nan.move_to_end(key)
        if len(self.cache_nan) > self.capacity:
            self.cache_nan.popitem(last=False)
