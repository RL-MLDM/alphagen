import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.loader import QlibDataLoader


def load_expr(expr: str, instrument: str, start_time: str, end_time: str) -> pd.DataFrame:
    return (QlibDataLoader(config={"feature": [expr]})      # type: ignore
            .load(instrument, start_time, end_time))


def correlation(joined: pd.DataFrame):
    return joined["factor"].corr(joined["target"], method="spearman")


class Evaluation:
    instrument: str
    start_time: str
    end_time: str
    target: pd.Series

    def __init__(self, instrument: str, start_time: str, end_time: str):
        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time

        self.target = self._load('Ref($close,-20)/$close-1').iloc[:, 0].rename("target")

    def _load(self, expr: str) -> pd.DataFrame:
        return load_expr(expr, self.instrument, self.start_time, self.end_time)

    def evaluate(self, expr: str) -> float:
        # self._load(expr).set_axis(labels=["factor"], axis=1)
        factor = self._load(expr).set_axis(labels=['factor'], axis=1)
        joined = factor.join(self.target).groupby('datetime')
        return joined.apply(correlation).mean()             # type: ignore


if __name__ == '__main__':
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    ev = Evaluation('csi300', '2016-01-01', '2018-12-31')
    print(ev.evaluate('Add(Ref(Abs($low),-10),Div($high,$close))'))
