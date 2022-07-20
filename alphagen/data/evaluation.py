import pandas as pd
import torch

from plotly.graph_objs._figure import Figure
from qlib.data.dataset.loader import QlibDataLoader

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.data.stock_data import StockData
from alphagen.utils.correlation import batch_spearmanr


def _load_expr(expr: str, instrument: str, start_time: str, end_time: str) -> pd.DataFrame:
    return (QlibDataLoader(config={"feature": [expr]})      # type: ignore
            .load(instrument, start_time, end_time))


def _correlation(joined: pd.DataFrame):
    return joined["factor"].corr(joined["target"], method="spearman")


class Evaluation:
    instrument: str
    start_time: str
    end_time: str

    def __init__(self,
                 instrument: str,
                 start_time: str, end_time: str,
                 target: Expression,
                 device: torch.device = torch.device("cpu")):
        self.data = StockData(instrument, start_time, end_time, device=device)
        self._target = target.evaluate(self.data)

        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time

        # self.target = self._load('Ref($close,-20)/$close-1').iloc[:, 0].rename("target")

    def _load(self, expr: str) -> pd.DataFrame:
        return _load_expr(expr, self.instrument, self.start_time, self.end_time)

    def evaluate(self, expr: Expression) -> float:
        try:
            factor = expr.evaluate(self.data)
        except OutOfDataRangeError:
            return -1.
        target = self._target.clone()
        corrs = batch_spearmanr(factor, target)
        return corrs.mean().item()

    def performance_graph(self, expr: Expression) -> Figure:
        from qlib.contrib.report.analysis_model import model_performance_graph
        data = self.data
        df = data.make_dataframe([self._target, expr.evaluate(data)], ["label", "score"])
        fig = model_performance_graph(
            df, rank=True, graph_names=["group_return"], show_notebook=False)[0]
        return fig


if __name__ == '__main__':
    from alphagen.data.expression import *

    high = Feature(FeatureType.HIGH)
    low = Feature(FeatureType.LOW)
    close = Feature(FeatureType.CLOSE)

    target = Ref(close, -20) / close - 1
    expr = Ref(abs(low), 10) + high / close

    ev = Evaluation('csi300', '2016-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))
