import numpy as np
import pandas as pd
import file_read


#returns [-1,1] whether to buy,stay and sell respectively

def mov_avg(k,file_data):
	
	cols   = list(filter(lambda x: 'close' in x, file_data.columns))
	data = file_data[cols].values
	n_stocks = data.shape[1]
	n_days = data.shape[0]

	avg = np.zeros((n_days-k,n_stocks))
	for i in range(n_stocks):
		for j in range(n_days - k):
			avg[j,i] = np.sum(data[j:j+k,i])/k

	mov_avg = np.zeros((n_days-k,n_stocks))
	mov_avg = data[k:] - avg

	for i in xrange(n_stocks):
		mov_avg[:,i] = (mov_avg[:,i] - np.min(mov_avg[:,i]))/(np.max(mov_avg[:,i]) - np.min(mov_avg[:,i]))
	return mov_avg,avg

def macd(file_data):
	#ref investopedia
	cols   = list(filter(lambda x: 'close' in x, file_data.columns))
	data = file_data[cols].values
	n_stocks = data.shape[1]
	n_days = data.shape[0]

	_,avg26 = mov_avg(26,file_data)
	_,avg12 = mov_avg(12,file_data)
	avg = (avg26 + avg12[14:])/2

	_,signal_line = mov_avg(9,file_data)
	

	macd = np.zeros((n_days-26,n_stocks))
	macd = signal_line[17:] - avg

	for i in xrange(n_stocks):
		macd[:,i] = (macd[:,i] - np.min(macd[:,i]))/(np.max(macd[:,i]) - np.min(macd[:,i]))
	return macd
		
def rsi(file_data):
	cols = list(filter(lambda x: 'return' in x, file_data.columns))
	data = np.exp(file_data[cols].values) - 1
	n_stocks = data.shape[1]
	n_days = data.shape[0]

	rsi = np.zeros((n_days,n_stocks))
	
	for i in xrange(n_stocks):
		for j in xrange(n_days-14):
			avg_gain = 0
			avg_loss = 0
			
			for m in range(14):
				if data[j+m,i] > 0:
					avg_gain += data[j+m,i]

				if data[j+m,i] < 0:
					avg_loss += -data[j+m,i] 

			rsi[14+j,i] = 1.0-((1.0)/(1 + avg_gain/avg_loss)) if avg_loss != 0 else 0
	
	return rsi

