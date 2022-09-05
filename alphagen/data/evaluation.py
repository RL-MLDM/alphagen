import pandas as pd
import torch

from plotly.graph_objs._figure import Figure
from qlib.data.dataset.loader import QlibDataLoader

from alphagen.data.expression import *
from alphagen.utils.correlation import batch_spearmanr


class Evaluation:
    instruments: List[str]
    start_time: str
    end_time: str

    def __init__(self,
                 instruments: List[str],
                 start_time: str, end_time: str,
                 target: Expression,
                 device: torch.device = torch.device("cpu")):
        self.data = StockData(instruments, start_time, end_time, device=device)
        self._target = target.evaluate(self.data)

        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time

    def _load(self, expr: str) -> pd.DataFrame:
        return (QlibDataLoader(config={"feature": [expr]})      # type: ignore
                .load(self.instruments, self.start_time, self.end_time))

    def evaluate(self, expr: Expression) -> float:
        try:
            factor = expr.evaluate(self.data)
        except OutOfDataRangeError:
            return -1.
        corrs = batch_spearmanr(factor, self._target)
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

    inst = StockData.list_instruments("csi300", "2016-01-01", "2016-01-01", 0, 0)
    ev = Evaluation(inst, '2016-01-01', '2018-12-31', target)
    print(ev.evaluate(expr))
