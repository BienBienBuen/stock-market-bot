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

matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
matplotlib.style.use('ggplot')


def percentage_change(Open, Close):
    percent = ((Close - Open)/Open)*100
    return percent

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
"""
with open('stock_symbols.txt') as f:
    with open('stock_data.csv', 'w') as p:
        #Retrieve data for every single stock
        for data in f:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

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
   
    


aapl_hist_data = yf.download(tickers="GME", period="MAX", interval = "1d")
aapl_hist_data.drop(['High','Low','Adj Close','Volume'], axis=1)

percentage_col = [percentage_change(x, y) for x, y in zip(aapl_hist_data['Open'], aapl_hist_data['Close'])]
aapl_hist_data['delta p'] = percentage_col

y, x = np.histogram(percentage_col, bins=150, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
params = st.norm.fit(percentage_col)
arg = params[:-2]
loc = params[-2]
scale = params[-1]

# Calculate fitted PDF and error with fit in distribution using SSE
pdf = st.norm.pdf(x, loc=loc, scale=scale, *arg)
sse = np.sum(np.power(y - pdf, 2.0))

print(sse)
print(params)

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

#visualization
pdf = make_pdf(st.norm, params)
ax = pdf.plot(lw=2, label='PDF', legend=True)
plt.hist(aapl_hist_data['delta p'], color = 'blue', density = True, edgecolor = 'black', bins=100)

plt.xlim(-20, 20)

plt.show()

"""
np.savetxt('historical_data.txt', aapl_hist_data.values)
aapl_hist_data.info()
for label, content in aapl_hist_data.items():
    print(f'label: {label}')
    print(f'content: {content}', sep='\n')

for i in range (len(percentage_col)):
    percentage_col[i] = round(percentage_col[i], 2)

"""
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