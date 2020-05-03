# Packages
import pandas as pd
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt

# Data import
aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2006, 10, 1),
                          end=datetime.datetime(2012, 1, 1))

# Data storage (temp)
aapl.to_csv('data/aapl_ohlc.csv')

# Data reading
df_aapl = pd.read_csv('data/aapl_ohlc.csv',
                      header=0,
                      index_col='Date',
                      parse_dates=True)

# Visualization
aapl['Close'].plot(grid=True)
plt.show()