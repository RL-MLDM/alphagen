from typing import List, Tuple, Union, Optional, ClassVar
from enum import IntEnum
import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    # VWAP = 5


class StockData:
    _qlib_initialized: ClassVar[bool] = False

    def __init__(
        self,
        instruments: List[str],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.init_qlib()

        self._instruments = instruments
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self._device = device
        self.data, self._dates, self._stock_ids = self._get_data()

    @property
    def device(self) -> torch.device: return self._device

    @classmethod
    def list_instruments(
        cls,
        instruments: str,
        start_time: str,
        end_time: str,
        backtrack_days: int = 100,
        future_days: int = 30
    ) -> List[str]:
        cls.init_qlib()
        from qlib.data import D
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(start_time))    # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(end_time))        # type: ignore
        real_start_time = cal[start_index - backtrack_days]
        if cal[end_index] != pd.Timestamp(end_time):
            end_index -= 1
        real_end_time = cal[end_index + future_days]
        inst = D.instruments(market=instruments)
        return D.list_instruments(inst, real_start_time, real_end_time, as_list=True)

    @classmethod
    def init_qlib(cls, data_provider_uri: Optional[str] = None) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        if data_provider_uri is None:
            data_provider_uri = "~/.qlib/qlib_data/cn_data"
        qlib.init(provider_uri=data_provider_uri, region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))      # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_time = cal[end_index + self.max_future_days]
        return D.features(self._instruments, exprs,
                          start_time=real_start_time, end_time=real_end_time)

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = df.stack().unstack(level=0)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self._device), dates, stock_ids

    @property
    def n_features(self) -> int: return len(self._features)

    @property
    def n_stocks(self) -> int: return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `columns`: an optional list of column names
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
