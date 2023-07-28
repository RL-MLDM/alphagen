from math import isnan

import pandas as pd
from alphagen.trade.base import StockPosition, StockStatus
from alphagen_qlib.calculator import QLibStockDataCalculator

from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_alpha_pool_by_path, load_recent_data


POOL_PATH = '/DATA/xuehy/logs/kdd_csi300_20_4_20230410071036/301056_steps_pool.json'


if __name__ == '__main__':
    data, latest_date = load_recent_data(instrument='csi300', window_size=365, offset=1)
    calculator = QLibStockDataCalculator(data=data, target=None)
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)

    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    df = data.make_dataframe(ensemble_alpha)

    strategy = TopKSwapNStrategy(K=20,
                                 n_swap=10,
                                 signal=df, # placeholder
                                 min_hold_days=1)

    signal = df.xs(latest_date).to_dict()['0']
    status = StockStatus(pd.DataFrame.from_records([
        (k, not isnan(v), not isnan(v), v) for k, v in signal.items()
    ], columns=['code', 'buyable', 'sellable', 'signal']))
    position = pd.DataFrame(columns=StockPosition.dtypes.keys()).astype(
        {col: str(dtype) for col, dtype in StockPosition.dtypes.items()}
    )

    to_buy, to_sell = strategy.step_decision(status_df=status,
                                             position_df=position)

    for i, code in enumerate(to_buy):
        if (i + 1) % 4 == 0:
            print(code)
        else:
            print(code, end=' ')
