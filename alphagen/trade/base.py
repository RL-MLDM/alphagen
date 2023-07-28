from enum import IntEnum
from typing import Dict, NamedTuple, Optional, Type

import pandera as pa


Amount = float
Price = float

StockCode = str
StockSignal = float

StockPosition = pa.DataFrameSchema({
    'code': pa.Column(StockCode),
    'amount': pa.Column(Amount),
    'days_holded': pa.Column(int),
})

StockStatus = pa.DataFrameSchema({
    'code': pa.Column(StockCode),
    'buyable': pa.Column(bool),
    'sellable': pa.Column(bool),
    'signal': pa.Column(StockSignal, nullable=True),
})


class StockOrderDirection(IntEnum):
    BUY = 1
    SELL = 2


class StockOrder:
    code: StockCode
    amount: Amount
    direction: Optional[StockOrderDirection]

    def __init__(self,
                 code: StockCode,
                 amount: Amount):
        self.code = code
        self.amount = amount
        self.direction = None

    def to_buy(self):
        self.direction = StockOrderDirection.BUY

    def to_sell(self):
        self.direction = StockOrderDirection.SELL

    def set_direction(self, direction: StockOrderDirection):
        self.direction = direction
