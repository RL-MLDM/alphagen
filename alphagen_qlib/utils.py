import datetime
import json
from typing import List, Tuple
from alphagen.data.expression import *
from alphagen_generic.features import *

from alphagen_qlib.stock_data import StockData
from alphagen.data.parser import ExpressionParser


def load_recent_data(instrument: str,
                     window_size: int = 365,
                     offset: int = 1,
                     **kwargs) -> Tuple[StockData, str]:
    today = datetime.date.today()
    start_date = str(today - datetime.timedelta(days=window_size))
    end_date = str(today - datetime.timedelta(days=offset))

    return StockData(instrument=instrument,
                     start_time=start_date,
                     end_time=end_date,
                     max_future_days=0,
                     **kwargs), end_date


def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    parser = ExpressionParser(Operators)
    exprs = [parser.parse(e) for e in raw["exprs"]]
    weights = raw["weights"]
    return exprs, weights


def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
        return load_alpha_pool(raw)
