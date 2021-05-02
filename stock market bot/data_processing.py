import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import string
import requests
import os
import csv
import warnings
from bs4 import BeautifulSoup

from numpy import log as ln
from numpy import random
from scipy.stats import binom

matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
matplotlib.style.use('ggplot')
# Global list of the tickers
L = ["AAPL","TSLA","QCOM","GME","AMC"]

def percentage_change(Open, Close):
    percent = ((Close - Open)/Open)*100
    return percent

def change(Open, Close):
    return Close/Open

def make_pdf(dist, params, size=100):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.0001, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.9999, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def t(x, u, d, N):
    for i in range (len(x)):
        x[i] = (ln(x[i])-N*ln(d))/(ln(u)-ln(d))
    return x


def partial(p1, p2, a1, a2, b1, b2):
    pass

#checking the if the file is empty
if os.stat('stock_symbols.txt').st_size==0:
#Downloading symbols of all stocks from the website
    alpha = list(string.ascii_uppercase)

    symbols = []

    for each in alpha:
        url = 'http://eoddata.com/stocklist/NYSE/{}.htm'.format(each)
        resp = requests.get(url)
        site = resp.content
        soup = BeautifulSoup(site, 'html.parser')
        table = soup.find('table', {'class': 'quotes'})
        for row in table.findAll('tr')[1:]:
            symbols.append(row.findAll('td')[0].text.rstrip())
        
    # Remove the extra letters on the end
    symbols_clean = []

    for each in symbols:
        each = each.replace('.', '-')
        symbols_clean.append((each.split('-')[0]))    

    #writing the symbols to a txt file
    np.savetxt('stock_symbols.txt', np.array(symbols_clean), fmt="%s")
else:
    pass

#IA specific, downloading data for stock tickers in L
# temp is data for all the stocks over a fixed period
# temp is for 6d 1m interval
if os.stat('IA_data_1w_1m.csv').st_size==0:
    for data in L:
        temp = yf.download(tickers=data, period="7d", interval = "1m")
        temp.drop(['High','Low','Adj Close','Volume'], axis=1)
        temp['delta p'] = [percentage_change(x, y) for x, y in zip(temp['Open'], temp['Close'])]
        temp.to_csv('IA_data_1w_1m.csv', mode='a', encoding='utf-8')
else:
    pass
#temp 2 is for 1mo 5m interval
if os.stat('IA_data_1mo_5m.csv').st_size==0:
    for data in L:
        temp2 = yf.download(tickers=data, period="1mo", interval = "5m")
        temp2.drop(['High','Low','Adj Close','Volume'], axis=1)
        temp2['delta p'] = [percentage_change(x, y) for x, y in zip(temp2['Open'], temp2['Close'])]
        temp2.to_csv('IA_data_1mo_5m.csv', mode='a', encoding='utf-8')
else:
    pass



#Modelling using binomial distribution
hist_data = yf.download(tickers="GME", start="2018-02-10", end="2021-02-09", interval = "1d").drop(['High','Low','Adj Close','Volume'], axis=1) 
hist_data2 = yf.download(tickers="LTC-USD", start="2018-02-10", end="2021-02-09", interval = "1d").drop(['High','Low','Adj Close','Volume'], axis=1)

#GME
change_col = [change(x, y) for x, y in zip(hist_data['Open'], hist_data['Close'])]
hist_data['X'] = change_col
sorted_col = np.sort(change_col, kind = 'mergesort')

#Litecoin
change_col2 = [change(x, y) for x, y in zip(hist_data2['Open'], hist_data2['Close'])]
hist_data2['Y'] = change_col2
sorted_col2 = np.sort(change_col2, kind = 'mergesort')

#values for u and d 
N = 390
d_i = sorted_col[0]
u_i =sorted_col[-1]
d = sorted_col[0]**(1/N)
u = sorted_col[-1]**(1/N)
print(d)
print(u)

#litecoin calculation only
d_i2 = sorted_col2[0]
u_i2 =sorted_col2[-1]
d2 = sorted_col2[0]**(1/N)
u2 = sorted_col2[-1]**(1/N)
print(d2)
print(u2)

#normal distribution graph
y, x = np.histogram(change_col, bins=300, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
params = st.norm.fit(change_col)
arg = params[:-2]
loc = params[-2]
scale = params[-1]

# Calculate fitted PDF and error with fit in distribution using SSE
pdf = st.norm.pdf(x, loc=loc, scale=scale, *arg)
sse = np.sum(np.power(y - pdf, 2.0))
print(sse)
print(params)

#visualization
pdf = make_pdf(st.norm, params)
ax = pdf.plot(lw=2, label='PDF', legend=True)
plt.hist(change_col, color = 'blue', density = True, edgecolor = 'black', bins=100)
plt.xlim(0.5, 1.5)
plt.show()

bin = []
for i in range(N+1):
    bin.append(i)
#Untransformed binomial distribution graph
plt.hist(change_col, color = 'blue', density = True, stacked = True, edgecolor = 'black', bins=300)
plt.xlabel("size the price is multiplied by")
plt.ylabel("probability density")
plt.title("Probability distribution of the size of the price multiplier of one trading day")

plt.xlim(0.5, 1.5)

plt.show()
#GME: transformed binomial distribution graph
t(change_col, u, d, N)
mean = np.mean(change_col)
prob = mean/N
print(prob)
#Kelly fraction
f = prob/(1-d) + (1-prob)/(1-u)
print(f)

#Litecoin: transformed binomial distribution graph
t(change_col2, u2, d2, N)
mean2 = np.mean(change_col2)
prob2 = mean2/N
print(prob2)
#Kelly fraction
f2 = prob2/(1-d2) + (1-prob2)/(1-u2)
print(f2)

# transformed histogram
plt.hist(change_col, color = 'blue', density = True, stacked = True, edgecolor = 'black', bins=bin)
plt.xlabel("number of successes")
plt.ylabel("probability density")
plt.title("Probability distribution of the number of successes in one trading day")
plt.xlim(0, 390)
plt.show()

# defining list of r values
r_values = list(range(N + 1))
# list of pmf values
dist = [binom.pmf(r, N, prob) for r in r_values]
# plotting the graph 
plt.bar(r_values, dist, width= 1)
plt.xlabel("number of successes")
plt.ylabel("probability density")
plt.title("Binomial distribution")
plt.xlim(0, 390)
plt.show()


#Partial w.r.t. y coefficients
print(prob*prob2*(u2-1))
print((1-prob)*prob2*(u2-1))
print(prob*(1-prob2)*(d2-1))
print((1-prob)*(1-prob2)*(d2-1))
#Partial w.r.t. x coefficients
print(prob*prob2*(u-1))
print((1-prob)*prob2*(d-1))
print(prob*(1-prob2)*(u-1))
print((1-prob)*(1-prob2)*(d-1))
#data for graphing 3d function 
print(prob*prob2)
print((1-prob)*prob2)
print(prob*(1-prob2))
print((1-prob)*(1-prob2))
print(u-1)
print(d-1)
print(u2-1)
print(d2-1)

#covariance part
# Matching the dates 
for i in range (len(hist_data)-3):
    while hist_data.index[i] > hist_data2.index[i]:
        hist_data2 = hist_data2.drop(hist_data2.index[i])
    while hist_data.index[i] < hist_data2.index[i]:    
        hist_data = hist_data.drop(hist_data.index[i])

print(hist_data)
print(hist_data2)

X = [change(x, y) for x, y in zip(hist_data['Open'], hist_data['Close'])]
Y = [change(x, y) for x, y in zip(hist_data2['Open'], hist_data2['Close'])]
XY = np.stack((X, Y), axis=0)
#variance for X
M = np.cov(XY)
var_x = M[0, 0]
#Mean for x
mean = np.mean(X)
print(M)
# for final section , sigma^2 and mu
print(var_x)
print(mean-1)

# covarince for X' and Y'
t(X, u, d, N)
t(Y, u2, d2, N)
X_pY_p = np.stack((X, Y), axis=0)
#Cov
#Bilinearity of covariance
cov_M = np.cov(X_pY_p)/(N**2)
cov_x_y = cov_M[0, 1]
print(cov_x_y)
#calculating a, b, c, and d
a = cov_x_y + prob*prob2
c = prob - a
b = prob2 - a
dd = 1-a-b-c
print(a)
print(b)
print(c)
print(dd)
#Partial w.r.t. y coefficients (with covariance)
print(a*(u2-1))
print(b*(u2-1))
print(c*(d2-1))
print(dd*(d2-1))
#Partial w.r.t. x coefficients (with covariance)
print(a*(u-1))
print(b*(d-1))
print(c*(u-1))
print(dd*(d-1))
plt.scatter(X, Y)
plt.xlabel("Gamestop")
plt.ylabel("Litecoin")
plt.show()


"""
np.savetxt('historical_data.txt', hist_data.values)
hist_data.info()
for label, content in hist_data.items():
    print(f'label: {label}')
    print(f'content: {content}', sep='\n')

for i in range (len(percentage_col)):
    percentage_col[i] = round(percentage_col[i], 2)

"""
#Reading data from stored txt file to start processing
"""
with open('historical_data.txt') as f:
    pass
    data = f.read()
    print(data)
"""
"""
1.For all strategies, do a 2 month 5m and 7 days 1m list 
2. Update historical_data.txt everyday
3. A system of reading and writing from the txt file
"""
"""
    f = open("IA_data_1mo_5m.csv", "w")
    f.truncate()
    f.close()
"""

"""
#compare against a proper binomial distribution 
mean = np.mean(change_col)
prob = mean/N
fig, ax = plt.subplots(1, 1)
n, p = N, prob
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
x = np.arange(binom.ppf(0.00000000001, n, p),
              binom.ppf(0.99999999999, n, p))
ax.plot(x, binom.pmf(x, n, p), 'bo', ms=1, label='binom pmf')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
plt.xlim(0, 390)

plt.legend()
plt.show()
"""

#Go through every single stock and calculate their percentage of increasing
"""
with open('stock_symbols.txt') as f:
    with open('stock_data.csv', 'w') as p:
        #Retrieve data for every single stock
        for data in f:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                # temp is data for all the stocks over a fixed period 
                temp = yf.download(tickers=data, period="7d", interval = "1m")
                temp.drop(['High','Low','Adj Close','Volume'], axis=1)
                temp['delta p'] = [percentage_change(x, y) for x, y in zip(temp['Open'], temp['Close'])]
            #Calculate pdf for every stock 
            #Requires try component   
                val = st.norm.fit(temp) #data contains non-finite value?
                params = [str(data)]
                for i in range(len(val)):
                    params.append(round(val[i], 9))
                csvwriter = csv.writer(p)
                csvwriter.writerow(params)
            
            except Exception:
                pass
"""