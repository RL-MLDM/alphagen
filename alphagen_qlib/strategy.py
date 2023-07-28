import copy
from math import isnan
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from alphagen.trade.base import StockCode, StockPosition, StockSignal, StockStatus

from alphagen.trade.strategy import Strategy


class TopKSwapNStrategy(BaseSignalStrategy, Strategy):
    def __init__(
        self,
        K, n_swap,
        min_hold_days=1,
        only_tradable=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.n_swap = n_swap
        self.min_hold_days = min_hold_days
        self.only_tradable = only_tradable

    def step_decision(self,
                      status_df: pd.DataFrame,
                      position_df: Optional[pd.DataFrame] = None
                     ) -> Tuple[List[StockCode], List[StockCode]]:
        signal = dict(zip(status_df['code'], status_df['signal']))
        unbuyable = set(record['code'] for record in status_df.to_dict('records') if not record['buyable'])
        unsellable = set(record['code'] for record in status_df.to_dict('records') if not record['sellable'])

        if position_df is None:
            days_holded = dict()
        else:
            days_holded = dict(zip(position_df['code'], position_df['days_holded']))

        all_valid_stocks = set(k for k, v in signal.items() if not isnan(v))
        all_holding_stocks = days_holded.keys()
        valid_holding_stocks = all_holding_stocks & all_valid_stocks

        n_to_open = self.K - len(days_holded)
        not_holding_stocks = all_valid_stocks - valid_holding_stocks

        to_buy, to_sell, to_open = [], [], []

        holding_priority = [] # All sellable stocks in descending order
        for stock_id in sorted(valid_holding_stocks, key=signal.get, reverse=True):
            if stock_id in unsellable:
                continue
            if days_holded[stock_id] < self.min_hold_days:
                continue
            holding_priority.append(stock_id)

        for stock_id in sorted(not_holding_stocks, key=signal.get, reverse=True):
            if stock_id in unbuyable:
                continue

            can_swap = len(to_buy) >= self.n_swap and holding_priority and signal[stock_id] > signal[holding_priority[-1]]
            if can_swap:
                to_sell.append(holding_priority.pop())
                to_buy.append(stock_id)
            elif len(to_open) < n_to_open:
                to_open.append(stock_id)
            else:
                break

        return to_buy + to_open, to_sell

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        time_per_step = self.trade_calendar.get_freq()
        current_temp = copy.deepcopy(self.trade_position)
        cash = current_temp.get_cash()

        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        stock_signal: StockSignal = pred_score.to_dict()

        def get_holding(stock_id: str):
            amount = None # Not required yet
            days_holded = current_temp.get_stock_count(stock_id, bar=time_per_step)
            return dict(code=stock_id, amount=amount, days_holded=days_holded)

        position = pd.DataFrame.from_records([
            get_holding(stock_id) for stock_id in current_temp.get_stock_list()
        ], columns=['code', 'days_holded'])

        def get_status(stock_id: str):
            if isnan(stock_signal[stock_id]):
                return dict(code=stock_id, signal=np.nan, buyable=False, sellable=False)
            buyable = self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY
            )
            sellable = self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL
            )
            return dict(code=stock_id, signal=stock_signal[stock_id], buyable=buyable, sellable=sellable)

        status = pd.DataFrame.from_records([get_status(code) for code in stock_signal])

        to_buy, to_sell = self.step_decision(status_df=status, position_df=position)

        buy_orders, sell_orders = [], []

        for code in to_sell:
            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_orders.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order, position=current_temp
                )
                cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(to_buy) if len(to_buy) > 0 else 0

        for code in to_buy:
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_orders.append(buy_order)

        return TradeDecisionWO(sell_orders + buy_orders, self)
