import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import time
import datetime
import file_read
import indicators


P = file_read.read_data()
target_cols = list(filter(lambda x: 'return' in x, P.columns))
open_cols   = list(filter(lambda x: 'open' in x, P.columns))
h_2_o_cols  = list(filter(lambda x: 'vol' in x, P.columns))
stat_cols  = list(filter(lambda x: 'close' in x, P.columns))

InputDF = P[open_cols].values
shape   = InputDF.shape
InputDF = np.append(InputDF,P[h_2_o_cols].values)
InputDF = np.append(InputDF,P[stat_cols].values)

InputDF  =  InputDF.reshape((3,shape[0],shape[1]))
InputDF = np.transpose(InputDF, [1,2,0])
TargetDF = P[target_cols]


input_data = InputDF
output_data = TargetDF.values

def data_i(x,length):
    res = np.zeros((x.shape[0]-length,length,x.shape[1],x.shape[2]))
    for i in xrange(x.shape[0]-length):
        new_data = x[i:i+length,:,:]
        res[i] = new_data
    return res

def data_o(x,length):
    res = np.zeros((x.shape[0]-length,length,x.shape[1]))
    for i in xrange(x.shape[0]-length):
        new_data = x[i:i+length,:]
        res[i] = new_data
    return res

sequence_length = 30
input_data = data_i(input_data,sequence_length)
output_data = data_o(output_data,sequence_length)
input_open = input_data[-2:,:,0,0]
returns = P[target_cols].values
returns = returns[-400:,4]
K = 10#Moving averages span 
lag = 1000

def mvg_avg(returns,K):
	avg = np.zeros((len(returns) - K))
	for i in xrange(len(returns) - K):
		avg[i] = np.sum(returns[i:i+K])/K
	return avg


# def auto_corr(returns,k):
# 	n = len(returns) #total number of points
# 	avg = np.mean(returns)
# 	var = np.var(returns)
# 	auto_corr = np.zeros(k)
# 	for i in xrange(1,k):		
# 		corr = 0
# 		for t in xrange(n-i):
# 			corr += ((returns[t] - avg)*(returns[t+i] - avg))
# 		auto_corr[i-1] = corr/(var*(n))
# 	return auto_corr


open_cols   = list(filter(lambda x: 'o1_o0' in x, P.columns))
open_data = P[open_cols].values
for i in xrange(319):
    open_data[:,i] = (open_data[:,i] - np.min(open_data[:,i]))/(np.max(open_data[:,i]) - np.min(open_data[:,i]))
vol_cols    = list(filter(lambda x: 'vol' in x, P.columns))
vol_data = P[vol_cols].values
for i in xrange(319):
    vol_data[:,i] = (vol_data[:,i] - np.min(vol_data[:,i]))/(np.max(vol_data[:,i]) - np.min(vol_data[:,i]))
return_cols = list(filter(lambda x: 'return' in x, P.columns))
return_data = P[return_cols].values


mov_avg,_ = indicators.mov_avg(K,P)
rsi= indicators.rsi(P)
# high = input_data[:,-1,50,1]
# diff= input_data[:,-1,3,2]
# mov_avg = indicators.macd(P)
# auto_corr = auto_corr(returns,lag)
# print np.any(np.isnan(mov_avg))
a = 3

result = adfuller(vol_data[:,a])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# plot_acf(high)
# plt.plot(mov_avg[:,0],'r')
# plt.plot(signal_line[:,0],'g')
# plt.plot(diff)
plt.plot(mov_avg[:,a],'b')
plt.plot(rsi[:,a],'g')
plt.plot(open_data[:,a],'r')
plt.plot(vol_data[:,a],'c')
plt.plot(np.zeros(rsi.shape))
plt.show()





#date analysis
#######################
# dates = [int(time.mktime(datetime.datetime.strptime(s, "%d-%B-%Y").timetuple())) for s in small_date]

# dates.sort()

# dates = [datetime.datetime.fromtimestamp(s).strftime('%d-%B-%Y') for s in dates if s<1483122600 and s>1451500200]

# print dates
# # print time.mktime(datetime.datetime.strptime("31-December-2015", "%d-%B-%Y").timetuple())
#########################



