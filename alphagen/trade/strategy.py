from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd

from alphagen.trade.base import StockCode


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def step_decision(self,
                      status_df: pd.DataFrame,
                      position_df: Optional[pd.DataFrame] = None
                     ) -> Tuple[List[StockCode], List[StockCode]]:
        pass
