# Packages
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas.core import datetools
import datetime
import statsmodels.api as sm

########
# BASICS
########

# Data import
aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2006, 10, 1),
                          end=datetime.datetime(2012, 1, 1))

# Add diff column to df
aapl['diff'] = aapl.Open - aapl.Close
# del aapl['diff'] # debug: uncomment to remove diff column

# Visualization check
aapl['Close'].plot(grid=True)
plt.show()

# Data storage (temp)
aapl.to_csv('data/aapl_ohlc.csv')

# Data reading
df_aapl = pd.read_csv('data/aapl_ohlc.csv',
                      header=0,
                      index_col='Date',
                      parse_dates=True)

# Daily change

## Setup
daily_close = aapl[['Adj Close']]
daily_pct_change = daily_close.pct_change()

## NA value handling
daily_pct_change.fillna(0, inplace=True)

## Inspect pct change
print(daily_pct_change)

## Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

## Inspect log pct change
print(daily_log_returns)

# Monthly and quarterly changes

## Resample to business months (BM)
monthly = aapl.resample("BM").apply(lambda x: x[-1])

## Monthly pct change
monthly_pct_change = monthly.pct_change()

## Resample to quarterly using mean value per quarter
quarterly = aapl.resample("4M").mean()

## Quarterly pct change
quarterly_pct_change = quarterly.pct_change()

# Plot daily_pct_change
daily_pct_change.hist(bins=50)
plt.show()

# Show summary statistics
print(daily_pct_change.describe())

# Cumulative daily rate of return
cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)

# Plot cum_daily_return
cum_daily_return.plot(figsize=(12,8))
plt.show()

# Cumulative monthly rate of return
cum_monthly_return = cum_daily_return.resample("M").mean()
print(cum_monthly_return)

###########
# PORTFOLIO
###########

# Ticker function setup
def get(tickers, startdate, enddate):
    def data(ticker):
        return(pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map (data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

# Ticker selection
tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']

# Get ticker data
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

# Transform and refocus to Adj Close
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate daily_pct_change
daily_pct_change = daily_close_px.pct_change()

# Plot distributions
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
plt.show()

# Scatter matrix
scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1, figsize=(12,12))
plt.show()

# Moving average of adjusted close
adj_close_px = aapl['Adj Close']
moving_avg_adj_close = adj_close_px.rolling(window=40).mean()
print(moving_avg_adj_close)

# Volatility
min_periods = 75
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
vol.plot(figsize=(10,8))
plt.show()

# Ordinary least-squares regression

## Select out Adj Close
all_adj_close = all_data[['Adj Close']]

## Calculate returns
all_returns = np.log(all_adj_close / all_adj_close.shift(1))

# Isolate AAPL returns
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# Isolate MSFT returns
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Combine AAPL and MSFT returns in new dataframe
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:] # Selection '[1:]' avoids NaN value
return_data.columns = ['AAPL', 'MSFT']

# Add constant
X = sm.add_constant(return_data['AAPL'])

# Construct model
model = sm.OLS(return_data['MSFT'], X).fit()
print(model.summary())

# Plot ordinary least squares
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
plt.grid(True)
plt.xlabel('AAPL Returns')
plt.ylabel('MSFT Returns')
plt.show()

# Plot rolling correlation
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()
plt.show()

##########################
# MOVING AVERAGE CROSSOVER
##########################

# Windows
short_window = 40
long_window = 100

# Dataframe initialization
signals = pd.DataFrame(index = aapl.index)
signals['signal'] = 0.0

# Short moving simple average
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Long moving simple average
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

# Generate trade orders
signals['positions'] = signals['signal'].diff()

# Plot signals
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Price in $')
aapl['Close'].plot(ax=ax1, color='r', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(signals.loc[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

#####################
# BACKTESTING: PANDAS
#####################

# Initial capital
initial_capital = float(100000.0)

# Positions dataframe initialization
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Portfolio initialization
portfolio = positions.multiply(aapl['Adj Close'], axis=0)

# Store difference in shares owned
pos_diff = positions.diff()

# Add holdings to portfolio
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()

# Add cash to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()

# Add total to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add returns to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# View portfolio head
print(portfolio.head())

# Visualize portfolio

## Create figure
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio Value in $')

## Plot equity curve
portfolio['total'].plot(ax=ax1, lw=2)
ax1.plot(portfolio.loc[signals.positions == 1.0].index,
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

######################
# BACKTESTING: ZIPLINE
######################

