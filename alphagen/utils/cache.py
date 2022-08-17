from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.best_key = ''
        self.best_value = -2.

    def __len__(self):
        return len(self.cache)

    def get(self, key: str) -> int:
        if key not in self.cache:
            return -2
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        if value > self.best_value:
            self.best_key = key
            self.best_value = value
