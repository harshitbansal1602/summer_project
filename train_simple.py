import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras import optimizers
from keras import losses
from keras.layers import Activation,Input,Conv2D,LSTM,Dense,Reshape,Dropout
import keras.backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import file_read
import indicators
from keras.initializers import Constant


def make_sequence(x,length):
    res = np.zeros((x.shape[0]-length,length,x.shape[1]))
    for i in xrange(x.shape[0]-length):
        new_data = x[i:i+length,:]
        res[i] = new_data
    return res

class Norm(Layer):

    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
    	ndim = K.ndim(x)
    	if ndim == 2:
        	return K.softmax

        elif ndim > 2:
        	e = K.relu(x) #relu
        	# e = K.exp(x)/(K.exp(x)+1) #sigmoid
        	s = K.sum(e, axis=self.axis, keepdims=True)
        	return e / s

    def compute_output_shape(self, input_shape):
        return input_shape

"""
Since the target in the model are the returns of the next day in respect to the inputs, the loss function should maximaize
the return of the last day in the sequence. When predicting the returns append data of only one new day at the end of sequence.
"""
def l_portfolio(y_true,y_pred):
    true_return = -(K.exp(y_true) - 1) # To maximize this
    total_return = tf.multiply(y_pred, true_return)
    loss = tf.reduce_mean(total_return,axis = 0)
    loss = tf.reduce_sum(loss[-1,:],axis = -1)
    return loss

def days_return(y_true,y_pred):
    pred_return = K.exp(y_true) - 1 
    total_return = tf.multiply(y_pred, pred_return)
    days_return = tf.reduce_sum(total_return[:,-1,:])
    return days_return*100

def d_return(y_true,y_pred):
    pred_return = np.exp(y_true) - 1 
    total_return = np.multiply(y_pred, pred_return)
    days_return = np.mean(total_return, axis = 0)
    days_return = np.sum(days_return[-1,:] ,axis=-1)
    return days_return*100

def normalize(x):
	for i in xrange(x.shape[1]):
		x[:,i] = (x[:,i] - np.min(x[:,i]))/(np.max(x[:,i]) - np.min(x[:,i]))
	return x

#getting data from csv files
file_data = file_read.read_data()

return_cols = list(filter(lambda x: 'return' in x, file_data.columns))
open_cols   = list(filter(lambda x: 'o1_o0' in x, file_data.columns))
vol_cols    = list(filter(lambda x: 'vol' in x, file_data.columns))

return_data = file_data[return_cols].values
open_data = normalize(file_data[open_cols].values)# normalizing between 0-1
vol_data = normalize(file_data[vol_cols].values)


n_train = -200 #number of training samples
n_test = int(n_train*(0.1)) #number of test samples
output_size = 319 #number of stocks
reg_str = 1.0 # regularization strength 
output_size_1 = output_size*2 # shape of filters in 3rd conv layer
output_size_2 = output_size # shape of filters in 3rd conv layer
output_size_3 = output_size # shape of filters in 3rd conv layer
hidden_size_lstm = 319 # 
batch_size = 30
sequence_length = 30 
adam = optimizers.Adam(lr = 1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

i_return = make_sequence(return_data,sequence_length)
i_open = make_sequence(open_data,sequence_length)
i_vol = make_sequence(vol_data,sequence_length)
i_macd = make_sequence(indicators.macd(file_data),sequence_length)
i_rsi = make_sequence(indicators.rsi(file_data),sequence_length)

x_open = Input(shape=(sequence_length,output_size))
x_vol = Input(shape=(sequence_length,output_size))
x_macd = Input(shape=(sequence_length,output_size))
x_rsi = Input(shape=(sequence_length,output_size))

returns = []


for k in range(2,10):
    data = keras.layers.Concatenate(axis=-1)([x_open,x_vol,x_macd,x_rsi]) ## concatinating for convolution

    output_dense_1 = Dense(output_size_1,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),bias_initializer=Constant(1))(data)
    output_dense_1 = Dropout(.4)(output_dense_1)

    output_dense_2 = Dense(output_size_2,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),bias_initializer=Constant(1))(output_dense_1)
    output_dense_2 = Dropout(.4)(output_dense_2)

    output_dense_3 = Dense(output_size_3,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),bias_initializer=Constant(1))(output_dense_2)

    portfolio = Norm(axis=-1,name='portfolio')(output_dense_3)

    l = k+1

    input_open  = i_open[n_train-l:-l]
    input_vol = i_vol[n_train-l:-l]
    input_macd = i_macd[n_train-l:-l]
    input_rsi = i_rsi[n_train-l:-l]
    input_return = i_return[n_train-l:-l]

    model = Model(inputs= [x_open,x_vol,x_macd,x_rsi],outputs = [portfolio])

    model.compile(loss= [l_portfolio], optimizer = adam, metrics = {'portfolio': days_return})

    model.fit(x=[input_open,input_vol,input_macd,input_rsi],y=[input_return],verbose=2,batch_size=batch_size,validation_split = 0.0,epochs=30)
    
    
    output_open = i_open[-k:-k+1]
    output_vol = i_vol[-k:-k+1]
    output_macd = i_macd[-k:-k+1]
    output_rsi = i_rsi[-k:-k+1]
    output_return = i_return[-k:-k+1]
    #intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('lstm').output)
    # intermediate_output = intermediate_layer_model.predict(x = [output_open,output_vol,output_macd,output_rsi])
    # print intermediate_output
    # print intermediate_output.shape
    outputs = model.predict(x = [output_open,output_vol,output_macd,output_rsi],verbose = 1,batch_size = 1)
    print d_return(output_return,outputs)
    returns.append(d_return(output_return,outputs))

print np.mean(returns)

plt.plot(outputs[0,-1,:])
plt.show()
plt.plot(returns)
plt.show()
np.savetxt("Return2.csv", returns, delimiter=",")