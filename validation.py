import json

from tqdm import tqdm

from alphagen.config import OPERATORS
from alphagen.data.expression import *
from alphagen.utils.random import reseed_everything
from alphagen_qlib.evaluation import QLibEvaluation
from assets import ZZ300_2019

FEATURES = {
    '$open': Feature(FeatureType.OPEN),
    '$close': Feature(FeatureType.CLOSE),
    '$high': Feature(FeatureType.HIGH),
    '$low': Feature(FeatureType.LOW),
    '$volume': Feature(FeatureType.VOLUME),
}

OPS = {op.__name__: op for op in OPERATORS}


def is_float_but_not_int(raw: str) -> bool:
    try:
        float(raw)
    except ValueError:
        return False
    try:
        int(raw)
        return False
    except ValueError:
        return True


def build_expr_tree(raw: str) -> Expression:
    if raw[0] == '$':
        return FEATURES[raw]
    elif is_float_but_not_int(raw):
        return Constant(float(raw))
    elif raw[0].isalpha():
        delimiter_pos = []
        level = 0
        for i, c in enumerate(raw):
            if c == '(':
                if level == 0:
                    delimiter_pos.append(i)
                level += 1
            elif c == ')':
                if level == 1:
                    delimiter_pos.append(i)
                level -= 1
            elif c == ',' and level == 1:
                delimiter_pos.append(i)
        op_raw = raw[:delimiter_pos[0]]
        args_raw = [raw[delimiter_pos[i] + 1: delimiter_pos[i+1]] for i in range(len(delimiter_pos) - 1)]
        op = OPS[op_raw]
        args = [build_expr_tree(t) for t in args_raw]
        return op(*args)
    else:
        return DeltaTime(int(raw))


FILENAME = '/DATA/xuehy/preload/zz300_static_20160101_20181231.json'
SAVE_TO = '/DATA/xuehy/preload/zz300_static_20190101_20211231.json'

if __name__ == '__main__':
    with open(FILENAME, encoding='utf-8') as f:
        cache = json.load(f)

    reseed_everything(0)
    device = torch.device("cpu")
    csi300_2019 = ZZ300_2019
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    ev = QLibEvaluation(
        instrument=csi300_2019,
        start_time='2019-01-01',
        end_time='2021-12-31',
        target=target,
        device=device
    )
    ev.cache.preload(SAVE_TO)

    keys = list(cache['valid'].keys()) + list(cache['nan'].keys())
    for i, key in enumerate(tqdm(keys), start=1):
        expr = build_expr_tree(key)
        assert str(expr) == key, (key, str(expr))
        ev.evaluate(expr)

        if i % 5000 == 0:
            ev.cache.save(SAVE_TO)

    ev.cache.save(SAVE_TO)
