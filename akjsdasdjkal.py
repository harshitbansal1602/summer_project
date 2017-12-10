import tensorflow as tf
import glob
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
import file_read
import indicators

P = file_read.read_data()
target_cols = list(filter(lambda x: 'return' in x, P.columns.values))
open_cols   = list(filter(lambda x: 'o1_o0' in x, P.columns.values))
h_2_o_cols  = list(filter(lambda x: 'h_2_o' in x, P.columns.values))
l_2_o_cols  = list(filter(lambda x: 'l_2_o' in x, P.columns.values))
zvol_cols    = list(filter(lambda x: 'zvol' in x, P.columns.values))

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
k = 5
output_channels = 1
kernel_size = [5,5]
lstm_high_hs = 20
lstm_low_hs = 20
hidden_size_lstm = 50
sequence_length = 15
hidden_size_ff = 319
batch_size = 30
output_size = hidden_size_ff
adam = optimizers.Adam(lr = .01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)


n_days = -200
n_odays = int(n_days*(0.1))
input_data = data_i(input_data,sequence_length)
output_data = data_o(output_data,sequence_length)

input_open = input_data[n_days:,:,:,0]
input_zvol = input_data[n_days:,:,:,3]
input_high = input_data[n_days:,:,:,1]
input_low = input_data[n_days:,:,:,2]
input_close= output_data[n_days:,:,:]
output_open = input_data[n_odays:,:,:,0]
output_zvol = input_data[n_odays:,:,:,3]

input_macd = data_o(indicators.macd(P),sequence_length)[n_days:]
_,input_mov_avg = indicators.mov_avg(k,P)
input_mov_avg = data_o(input_mov_avg,sequence_length)[n_days:]
input_obv = data_o(indicators.obv(P),sequence_length)[n_days:]
input_rsi = data_o(indicators.rsi(P),sequence_length)[n_days:]




class Add(Layer):

    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

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



def l_portfolio(y_true,y_pred):
    true_return = K.exp(y_true) - 1 #since we want to maximize this
    total_return = tf.multiply(y_pred, true_return)
    # print total_return.shape = [?,sequence_length,output_size]
    loss = tf.reduce_mean(total_return[:,-1,:])
    return loss

def days_return(y_true,y_pred):
    pred_return = 1 - K.exp(y_true)
    total_return = tf.multiply(y_pred, pred_return)
    days_return = tf.reduce_sum(total_return[:,-1,:])
    return days_return



#model
#x = Input(shape=(sequence_length,output_size,4))
x = Input(shape=(sequence_length,output_size))

zvol_x = Input(shape=(sequence_length,output_size))

open_high = LSTM(lstm_high_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_high = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),name='open_high')(open_high)

open_low = LSTM(lstm_low_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_low = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),name='open_low')(open_low)

data = keras.layers.Concatenate(axis=-1)([open_high,open_low,zvol_x]) #remove or keep x?

data = keras.layers.core.Reshape([sequence_length,output_size,3])(data)

output_conv = Conv2D(data_format="channels_last",filters = output_channels,kernel_regularizer = keras.regularizers.l2(1.0) ,kernel_size = kernel_size,padding = "same",activation = 'relu')(data)

input_lstm = Add(axis=-1)(output_conv)

output_lstm = LSTM(hidden_size_lstm,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(input_lstm)

# portfolio = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0))(output_lstm)
model_output = Dense(hidden_size_ff,activation = 'tanh',kernel_regularizer = keras.regularizers.l2(1.0))(output_lstm)

macd = Input(shape=(sequence_length,output_size))
mov_avg = Input(shape=(sequence_length,output_size))
obv = Input(shape=(sequence_length,output_size))
rsi = Input(shape=(sequence_length,output_size))

final_input = keras.layers.Concatenate(axis=-1)([model_output,macd,mov_avg,obv,rsi]) #remove or keep x?


final_output = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0))(final_input)


portfolio = Norm(axis=-1,name='portfolio')(final_output)


model = Model(inputs= [x,zvol_x,macd,mov_avg,obv,rsi],outputs = [open_high,open_low,portfolio])
model.compile(loss= ['mean_squared_error','mean_squared_error',l_portfolio], optimizer = adam,metrics ={'portfolio': days_return},loss_weights=[.25,.25,.1])
model.fit(x=[input_open,input_zvol,input_macd,input_mov_avg,input_obv,input_rsi],y=[input_high,input_low,input_close],verbose=2,batch_size=batch_size,shuffle = False,validation_split = 0.1,epochs=500)
model.save_weights('weights.h5')

