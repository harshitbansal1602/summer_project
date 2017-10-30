import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

columns = np.array(['Date','Close Price'])
files = np.array([])
all_sizes = np.array([])
all_data = pd.DataFrame()

print 'Reading data....'
files = glob.glob("csv_high_new/*.csv")
small_date = pd.read_csv(files[0], usecols=['Date']).values

for f in files:
    date = pd.read_csv(f, usecols=['Date']).values
    small_date = np.intersect1d(small_date, date)

def get_ticker(filepath):
  filepath = filepath[-10:-4]
  return filepath

def make_inputs(filepath):
    D = pd.read_csv(filepath, usecols=columns)
    #Load the dataframe with headers
    D = D.drop_duplicates(subset = ['Date'])
    D =  D.loc[D['Date'].isin(small_date)]
    D = D.reset_index(drop=True)
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)
    Res['c1_c0'] = D['Close Price']
    Res['ticker'] = ticker
    return Res

for f in files:
  Res = make_inputs(f)
  all_data = all_data.append(Res)

pivot_columns = all_data.columns[:-1]
P = all_data.pivot_table(index=all_data.index,columns='ticker',values=pivot_columns)
mi = P.columns.tolist()
new_ind = pd.Index(e[1] +'_' + e[0] for e in mi)
P.columns = new_ind
P = P.sort_index(axis=1,ascending=True) # Sort by columns

target_cols = list(filter(lambda x: 'c1_c0' in x, P.columns.values))

returns = P[target_cols].values
returns = returns[:,1]
K = 5#Moving averages span 
lag = 1000
def mvg_avg(returns,K):
	avg = np.zeros((len(returns) - K))
	for i in xrange(len(returns) - K):
		avg[i] = np.sum(returns[i:i+K])/K
	return avg


def auto_corr(returns,k):
	n = len(returns) #total number of points
	avg = np.mean(returns)
	var = np.var(returns)
	auto_corr = np.zeros(k)
	for i in xrange(1,k):		
		corr = 0
		for t in xrange(n-i):
			corr += ((returns[t] - avg)*(returns[t+i] - avg))
		auto_corr[i-1] = corr/(var*(n))
	return auto_corr




mvg_avg = mvg_avg(returns,K)
auto_corr = auto_corr(returns,lag)
plot_pacf(returns)
plt.show()






