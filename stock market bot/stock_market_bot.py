import numpy as np
import pandas as pd
import requests
import alpaca_trade_api as tradeapi
import data_processing
import time
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import datetime, timedelta
from pytz import timezone
from data_processing import f
api_time_format = '%Y-%m-%dT%H:%M:%S.%f-04:00'

class portfolio:
    def __init__ (self, f1, f2): #can use *args for larger portfolios 
        self.portfolio = {
            "safe" : 100000.0 ,
            "p1" : 0.0,
            "p2" : 0.0
            }

        self.value = self.portfolio["safe"] + self.portfolio["p1"]
        self.f = [f1, f2]# self.f = list().append(f for f in *kelly)
        self.portfolio["p1"] = f1*self.value
        self.portfolio["p2"] = f2*self.value
        self.portfolio["safe"] -= (self.portfolio["p1"]+self.portfolio["p2"])
        self.last_close = 1 #self.lastclose = list().append(0 for i in range (len(kelly)))
        self.last_close2 = 1

    def adjust(self, open, close, open2, close2): #can use *args for larger portfolios 
        # value change after one bet/timeframe
        self.portfolio["p1"] *= close/self.last_close
        if close2 != 0:
            self.portfolio["p2"] *= close2/self.last_close2
        # This is the update part, after recalculating the proportions
        val = (self.portfolio["p1"] + self.portfolio["p1"] + self.portfolio["safe"])
        self.portfolio["p1"] = self.f[0]*val
        self.portfolio["p2"] = self.f[1]*val
        self.portfolio["safe"] = (1 -self.f[1]-self.f[0])*val
        self.last_close = close
        self.last_close2 = close2
        self.value = val
        return self.value 

    def momentum_strat(self):
        pass

    def Volatility (self):
        pass


#Public and private key to access account
api = tradeapi.REST('PKRWAYC25MT5QMCVMEU3', '9vQ9DyjhDYy3MOrCArNmzZJPP84BftjYrXuZl1IW', base_url='https://paper-api.alpaca.markets') 
account = api.get_account()
positions = api.list_positions()

#Sending trades to alpaca api to execute order
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
        stop_price='100.0',
        limit_price='100.0',
    ))
"""

#One stock
bot = portfolio(0.24466950097405515, 0)
df = pd.read_csv('IA_data_1w_1m.csv')
gme = df[8189:10913].apply(pd.to_numeric, errors='ignore') #AAPL 1794, 2728, QCOM 5460 - 8186, GME 8189 -- 10913, GME month 5387 -- 7179
#Creating a value column for the value of the portfolio when kelly strategy is not applied
bot.last_close = gme['Open'][8189]
gme['pval'] = [(val/gme['Open'][8189])*bot.value for val in gme['Close']]
gme['pval'].plot(label='Value without optimization')

#Creating a value column for when the kelly strategy is applied
gme['val'] = [bot.adjust(open, close, 0, 0) for open, close in zip(gme['Open'], gme['Close'])]
gme['val'].plot(label='Value with optimization')


plt.xlabel("time(minutes)")
plt.ylabel("Value($)")
plt.title("Value of portfolio in investing in Gamestop stock")

plt.legend()
plt.show()

#two stocks
bot2 = portfolio(0.2466949, 0.55996150) #no cov
bot2_cov = portfolio(0.2445086, 0.5598149)#considers cov
gme_df = yf.download(tickers="GME", start="2021-04-26", end="2021-04-30", interval = "1m").apply(pd.to_numeric, errors='ignore')
lite_df = yf.download(tickers="LTC-USD", start="2021-04-26", end="2021-04-30", interval = "1m").apply(pd.to_numeric, errors='ignore')
#matching time for data
for i in range (len(gme_df)):
    while gme_df.index[i] > lite_df.index[i]:
        lite_df = lite_df.drop(lite_df.index[i])
#set up
bot2.last_close = gme_df['Open'][0]
bot2.last_close2 = lite_df['Open'][0]
bot2_cov.last_close = bot2.last_close
bot2_cov.last_close2 = bot2.last_close2 
#create value colomns (doesn't work yet)
gme['nocov']  = [bot2.adjust(open, close, open2, close2) for open, close, open2, close2 in zip(gme_df['Open'], gme_df['Close'], lite_df['Open'], lite_df['Close'])]
gme['cov'] = [bot2_cov.adjust(open, close, open2, close2) for open, close, open2, close2 in zip(gme_df['Open'], gme_df['Close'], lite_df['Open'], lite_df['Close'])]
#plotting
gme['nocov'].plot(label='Not considering covariance')
gme['cov'].plot(label='Considering covariance')
plt.legend()
plt.show()

