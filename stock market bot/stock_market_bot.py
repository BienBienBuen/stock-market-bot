import numpy as np
import pandas as pd
import requests
import alpaca_trade_api as tradeapi
import data_processing
import time

from datetime import datetime, timedelta
from pytz import timezone
api_time_format = '%Y-%m-%dT%H:%M:%S.%f-04:00'

class bots:
    def __init__ (self):
        pass

    def momentum_strat(self):
        pass

    def stupid(self):
        pass

    def kelly_strat(self):
        pass

api = tradeapi.REST('PKINQZPKOQAV61LVM8OE', 'HLf9tdXqnrMbSwpqDgQsXMl9ql7Da0jlwSP77zMd', base_url='https://paper-api.alpaca.markets') # or use ENV Vars shown below
account = api.get_account()
positions = api.list_positions()

"""
api.submit_order(symbol='QCOM',
    side='buy',
    type='market',
    qty='100',
    time_in_force='day',
    order_class='bracket',
    take_profit=dict(
        limit_price='305.0',
    ),
    stop_loss=dict(
        stop_price='160.0',
        limit_price='160.0',
    ))

"""


with open('stock_data.csv') as f:
    bot1=bots()
    bot1.kelly_strat



