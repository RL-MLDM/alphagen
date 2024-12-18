import os
import time
import shutil
import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import baostock as bs
from baostock.data.resultset import ResultData
from baostock_utils import baostock_login_context, baostock_relogin, baostock_login

from qlib_dump_bin import DumpDataAll


def _read_all_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write_all_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


class DataManager:
    _all_a_shares: List[str]
    _basic_info: pd.DataFrame
    _adjust_factors: pd.DataFrame

    _adjust_columns: List[str] = [
        "foreAdjustFactor",
        "backAdjustFactor",
        "adjustFactor"
    ]
    _fields: List[str] = [
        "date", "open", "high", "low",
        "close", "preclose", "volume", "amount",
        "turn", "tradestatus", "pctChg", "peTTM",
        "psTTM", "pcfNcfTTM", "pbMRQ", "isST"
    ]
    _price_fields: List[str] = [
        "open", "high", "low", "close", "preclose"
    ]

    def __init__(
        self,
        save_path: str,
        qlib_export_path: str,
        qlib_base_data_path: Optional[str],
        use_forward_adjust: bool = False,
        adjust_date: Optional[str] = None,
        max_workers: int = 40,
        max_retries: int = 20,
        retry_wait_seconds: float = 5.
    ):
        self._save_path = os.path.expanduser(save_path)
        self._export_path = f"{self._save_path}/export"
        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(self._export_path, exist_ok=True)
        self._qlib_export_path = os.path.expanduser(qlib_export_path)
        self._qlib_path = qlib_base_data_path
        if self._qlib_path is not None:
            self._qlib_path = os.path.expanduser(self._qlib_path)
        self._use_forward_adjust = use_forward_adjust
        self._adjust_type: str = "foreAdjustFactor" if use_forward_adjust else "backAdjustFactor"
        self._adjust_date = adjust_date
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._retry_wait_seconds = retry_wait_seconds

    @property
    def _a_shares_list_path(self) -> str:
        return f"{self._save_path}/a_shares_list.txt"

    def _load_all_a_shares_base(self) -> None:
        if os.path.exists(self._a_shares_list_path):
            lines = _read_all_text(self._a_shares_list_path).split('\n')
            self._all_a_shares = [line for line in lines if line != ""]
        elif self._qlib_path is not None:
            lines = _read_all_text(f"{self._qlib_path}/instruments/all.txt").split('\n')
            all_a_shares = [line.split('\t')[0] for line in lines if line != ""]
            self._all_a_shares = [f"{stk_id[:2].lower()}.{stk_id[-6:]}"
                                  for stk_id in all_a_shares]

    def _load_all_a_shares(self):
        print("Loading A-Shares stock list")
        with baostock_login_context():
            self._load_all_a_shares_base()
            def query(): return bs.query_all_stock(day=str(datetime.date.today()))
            stocks = {code for code in self._query_as_data_frame(query)["code"]
                      if code.startswith("sh") or code.startswith("sz")}
            self._all_a_shares += ["sh.000903", "sh.000300", "sh.000905", "sh.000852"]
            self._all_a_shares = list(set(self._all_a_shares).union(stocks))
            _write_all_text(self._a_shares_list_path,
                            '\n'.join(str(s) for s in self._all_a_shares))

    def _parallel_foreach(
        self,
        callable,
        input: List[dict],
        max_workers: Optional[int] = None,
        need_to_login_baostock: bool = False
    ) -> list:
        if max_workers is None:
            max_workers = self._max_workers
        with tqdm(total=len(input)) as pbar:
            results = []
            login = baostock_login if need_to_login_baostock else None
            with ProcessPoolExecutor(max_workers, initializer=login) as executor:
                futures = [executor.submit(callable, **elem) for elem in input]
                for f in as_completed(futures):
                    results.append(f.result())
                    pbar.update(n=1)
            return results

    def _fetch_basic_info_job(self, code: str) -> pd.DataFrame:
        return self._query_as_data_frame(lambda: bs.query_stock_basic(code))

    def _fetch_basic_info(self) -> pd.DataFrame:
        print("Fetching basic info")
        dfs = self._parallel_foreach(
            self._fetch_basic_info_job,
            [dict(code=code) for code in self._all_a_shares],
            need_to_login_baostock=True
        )
        df = pd.concat(dfs)
        df = df.sort_values(by="code").drop_duplicates(subset="code").set_index("code")
        df.to_csv(f"{self._save_path}/basic_info.csv")
        return df

    def _fetch_adjust_factors_job(self, code: str, start: str) -> pd.DataFrame:
        return self._query_as_data_frame(lambda: bs.query_adjust_factor(code, start))

    def _fetch_adjust_factors(self) -> pd.DataFrame:
        def one_year_before_ipo(ipo: str) -> str:
            earliest_time = pd.Timestamp("1990-12-19")
            ts = pd.Timestamp(ipo) - pd.DateOffset(years=1)
            ts = earliest_time if earliest_time > ts else ts
            return ts.strftime("%Y-%m-%d")

        print("Fetch adjust factors")
        dfs: List[pd.DataFrame] = self._parallel_foreach(
            self._fetch_adjust_factors_job,
            [dict(code=code, start=one_year_before_ipo(data["ipoDate"]))
             for code, data in self._basic_info.iterrows()],
            need_to_login_baostock=True
        )
        df = pd.concat([df for df in dfs if not df.empty])
        df = df.set_index(["code", "dividOperateDate"])
        df.to_csv(f"{self._save_path}/adjust_factors.csv")
        return df

    def _adjust_factors_for(self, code: str) -> pd.DataFrame:
        adj_factor_idx: pd.Index = self._adjust_factors.index.levels[0]     # type: ignore
        if code not in adj_factor_idx:
            start: str = self._basic_info.loc[code, "ipoDate"]              # type: ignore
            return pd.DataFrame(
                [[1., 1., 1.]],
                index=pd.Index([start]),
                columns=self._adjust_columns
            )
        return self._adjust_factors.xs(code, level="code").astype(float)    # type: ignore

    def _download_stock_data_job(self, code: str, data: pd.Series) -> None:
        fields_str = ",".join(self._fields)
        numeric_fields = self._fields.copy()
        numeric_fields.pop(0)

        adj = self._adjust_factors_for(code)

        def query():
            return bs.query_history_k_data_plus(
                code, fields_str,
                start_date=data["ipoDate"],
                adjustflag=("2" if self._use_forward_adjust else "1")
            )
        res = self._query_as_data_frame(query)
        try:
            df = res.join(adj, on="date", how="left")
        except:
            print(f"{code = }\n{query = }\n{res = }")
            exit(1)
        df[self._adjust_columns] = df[self._adjust_columns].fillna(method="ffill").fillna(1.)
        df[numeric_fields] = df[numeric_fields].replace("", "0.").astype(float)

        def as_of_date(df: pd.DataFrame, date: str) -> pd.Series:
            index: int = df.index.searchsorted(date, side="right") - 1   # type: ignore
            return df.iloc[index]

        if self._adjust_date is not None:
            ref_factor = as_of_date(df, self._adjust_date)[self._adjust_type]
            readjust_fields = self._price_fields + [self._adjust_type]
            df[readjust_fields] /= ref_factor
        df["volume"] /= df[self._adjust_type]
        df["vwap"] = df["amount"] / df["volume"]
        df = df.set_index("date")
        df.to_pickle(f"{self._save_path}/k_data/{code}.pkl")

    def _download_stock_data(self) -> None:
        print("Download stock data")
        os.makedirs(f"{self._save_path}/k_data", exist_ok=True)
        self._parallel_foreach(
            self._download_stock_data_job,
            [dict(code=code, data=data)
             for code, data in self._basic_info.iterrows()],
            need_to_login_baostock=True
        )

    def _save_csv_job(self, path: Path) -> None:
        code = path.stem
        code = f"{code[:2].upper()}{code[-6:]}"
        df: pd.DataFrame = pd.read_pickle(path)
        df.rename(columns={self._adjust_type: "factor"}, inplace=True)
        df["code"] = code
        out = Path(self._export_path) / f"{code}.csv"
        df.to_csv(out)

    def _save_csv(self) -> None:
        print("Export to csv")
        children = list(Path(f"{self._save_path}/k_data").iterdir())
        self._parallel_foreach(
            self._save_csv_job,
            [dict(path=path) for path in children]
        )

    def _query_as_data_frame(self, query: Callable[[], ResultData]) -> pd.DataFrame:
        retries = 0
        while True:
            rows = []
            result = query()
            while result.error_code == "0":
                if not result.next():
                    return pd.DataFrame(rows, columns=result.fields)
                rows.append(result.get_row_data())
            retries += 1
            if retries > self._max_retries:
                msg = (f"Retry attempts exceeds the limit of {self._max_retries}, "
                       f"error code: {result.error_code}, error message: {result.error_msg}")
                print(msg)
                raise Exception(msg)
            time.sleep(self._retry_wait_seconds)
            baostock_relogin()

    def _dump_qlib_data(self) -> None:
        DumpDataAll(
            csv_path=self._export_path,
            qlib_dir=self._qlib_export_path,
            max_workers=self._max_workers,
            exclude_fields="date,code",
            symbol_field_name="code"
        ).dump()
        shutil.copy(f"{self._qlib_export_path}/calendars/day.txt",
                    f"{self._qlib_export_path}/calendars/day_future.txt")

    def _fix_constituents(self) -> None:
        today = str(datetime.date.today())
        path = f"{self._qlib_export_path}/instruments"

        for p in Path(path).iterdir():
            if p.stem == "all":
                continue
            df = pd.read_csv(p, sep='\t', header=None)
            df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)     # type: ignore
            latest_data = df[2].max()
            df[2] = df[2].replace(latest_data, today)
            df.to_csv(p, header=False, index=False, sep='\t')

    def fetch_and_save_data(
        self,
        use_cached_basic_info: bool = False,
        use_cached_adjust_factor: bool = False
    ):
        self._load_all_a_shares()
        if use_cached_basic_info:
            self._basic_info = pd.read_csv(f"{self._save_path}/basic_info.csv", index_col=0)
        else:
            self._basic_info = self._fetch_basic_info()
        if use_cached_adjust_factor:
            self._adjust_factors = pd.read_csv(f"{self._save_path}/adjust_factors.csv", index_col=[0, 1])
        else:
            self._adjust_factors = self._fetch_adjust_factors()
        self._download_stock_data()
        self._save_csv()
        self._dump_qlib_data()
        self._fix_constituents()


if __name__ == "__main__":
    dm = DataManager(
        save_path="../data",
        qlib_export_path="~/.qlib/qlib_data/cn_data_2024h1",
        qlib_base_data_path="~/.qlib/qlib_data/cn_data",
        adjust_date="2009-01-01"
    )
    dm.fetch_and_save_data()
    # dm._dump_qlib_data()
