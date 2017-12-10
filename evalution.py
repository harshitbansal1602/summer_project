import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras import optimizers
from keras import losses
from keras.layers import Activation,Input,Conv2D,LSTM,Add,Dense,TimeDistributed
import keras.backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import indicators
import file_read

P = file_read.read_data()


target_cols = list(filter(lambda x: 'return' in x, P.columns.values))
open_cols  = list(filter(lambda x: 'o1_o0' in x, P.columns.values))
h_2_o_cols  = list(filter(lambda x: 'h_2_o' in x, P.columns.values))
l_2_o_cols  = list(filter(lambda x: 'l_2_o' in x, P.columns.values))
zvol_cols    = list(filter(lambda x: 'vol' in x, P.columns.values))



InputDF = P[open_cols].values
shape   = InputDF.shape
InputDF = np.append(InputDF,P[h_2_o_cols].values)
InputDF = np.append(InputDF,P[l_2_o_cols].values)
InputDF = np.append(InputDF,P[zvol_cols].values)

InputDF  =  InputDF.reshape((4,shape[0],shape[1]))
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


#hyper parameters
k=10
output_channels = 1
kernel_size = [5,5]
lstm_high_hs = 20
lstm_low_hs = 20
hidden_size_lstm = 30
sequence_length = 30
hidden_size_ff = 319
batch_size = 20
output_size = hidden_size_ff
adam = optimizers.Adam(lr=10, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

n_days = -200
n_odays = int(n_days*(0.1))
input_data = data_i(input_data,sequence_length)
output_data = data_o(output_data,sequence_length)


class Add(Layer):

    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

class Norm(Layer):

    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
    	ndim = K.ndim(x)
    	if ndim == 2:
        	return K.softmax(x)
        elif ndim > 2:
        	e = K.relu(x,alpha=0.,max_value=None) #relu
        	# e = K.exp(x)/(K.exp(x)+1) #sigmoid
        	s = K.sum(e, axis=self.axis, keepdims=True)
        	return e / s

    def compute_output_shape(self, input_shape):
        return input_shape


def days_return(y_true,y_pred):
    pred_return = 1 - np.exp(y_true)
    total_return = np.multiply(y_pred, pred_return)
    days_return = np.sum(total_return[-1,-1,:],axis=-1)
    return days_return


#model
#x = Input(shape=(sequence_length,output_size,4))
x = Input(shape=(sequence_length,output_size))

zvol_x = Input(shape=(sequence_length,output_size))

open_high = LSTM(lstm_high_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_high = Dense(hidden_size_ff,activation = 'tanh',kernel_regularizer = keras.regularizers.l2(1.0),name='open_high')(open_high)

open_low = LSTM(lstm_low_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_low = Dense(hidden_size_ff,activation = 'tanh',kernel_regularizer = keras.regularizers.l2(1.0),name='open_low')(open_low)

macd = Input(shape=(sequence_length,output_size))
mov_avg = Input(shape=(sequence_length,output_size))
rsi = Input(shape=(sequence_length,output_size))

data = keras.layers.Concatenate(axis=-1)([open_high,open_low,zvol_x,macd,mov_avg,rsi])

data = keras.layers.core.Reshape([sequence_length,output_size,6])(data)

output_conv = Conv2D(data_format="channels_last",filters = output_channels,kernel_regularizer = keras.regularizers.l2(1.0) ,kernel_size = kernel_size,padding = "same",activation = 'relu')(data)

input_lstm = Add(axis=-1)(output_conv)

output_lstm = LSTM(hidden_size_lstm,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(input_lstm)

output_lstm = keras.layers.Dropout(.2)(output_lstm)


# portfolio = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0))(output_lstm)
model_output = Dense(hidden_size_ff,activation = 'tanh',kernel_regularizer = keras.regularizers.l2(1.0))(output_lstm)

# final_output = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0))(model_output)

portfolio = Norm(axis=-1,name='portfolio')(model_output)

model = Model(inputs= [x,zvol_x,macd,mov_avg,rsi],outputs = [open_high,open_low,portfolio])
model.load_weights('weights.h5')
returns = []

macd = data_o(indicators.macd(P),sequence_length)
mov_avg,_ = indicators.mov_avg(k,P)
mov_avg = data_o(mov_avg,sequence_length)
rsi = data_o(indicators.rsi(P),sequence_length)

for k in xrange(1,20):
    
    output_open = input_data[n_odays-k:-k,:,:,0]
    output_zvol = input_data[n_odays-k:-k,:,:,3]
    output_macd = macd[n_odays-k:-k]
    output_mov_avg = mov_avg[n_odays-k:-k]
    output_rsi = rsi[n_odays-k:-k]
    
    outputs = model.predict(x = [output_open,output_zvol,output_macd,output_mov_avg,output_rsi],verbose = 1)

    o_output = output_data[n_odays-k:-k,:,:].astype(np.float32)
    
    returns.append(days_return(o_output,outputs[2])*100)

    

def mvg_avg(returns,K):
	avg = np.zeros((len(returns) - K))
	for i in xrange(len(returns) - K):
		avg[i] = np.sum(returns[i:i+K])/K
	return avg




K = 10 #moving average constant

#mvg_avg = mvg_avg(returns,K)

# plt.plot(outputs[2][-1,-1,:])
# plt.plot(mvg_avg)
print np.mean(returns)
plt.plot(returns)
plt.show()
# np.savetxt("Return.csv", returns, delimiter=",")

