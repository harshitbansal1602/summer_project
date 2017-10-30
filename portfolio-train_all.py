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


columns = np.array(['Date','Open Price','High Price','Low Price','Close Price','No. of Trades'])
files = np.array([])
all_sizes = np.array([])
all_data = pd.DataFrame()

print 'Reading data....'
files = glob.glob("csv_high_new/*.csv")
small_date = pd.read_csv(files[0], usecols=['Date']).values
for f in files:
    date = pd.read_csv(f, usecols=['Date']).values
    small_date = np.intersect1d(small_date, date)
ret = lambda x,y: np.log(y/x) #Log return 
zscore = lambda x:(x -x.mean())/x.std() # zscore

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
    Res['open'] = ret(D['Open Price'],D['Open Price'].shift(-1)).fillna(0)
    Res['h_2_o'] = ret(D['Open Price'],D['High Price'])
    Res['l_2_o'] = ret(D['Open Price'],D['Low Price'])
    Res['c1_c0'] = ret(D['Close Price'],D['Close Price'].shift(-1)).fillna(0) #todays return
    Res['vol'] = zscore(D['No. of Trades'].shift(-1)).fillna(0)
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
print P
target_cols = list(filter(lambda x: 'c1_c0' in x, P.columns.values))
#c_2_o_cols  = list(filter(lambda x: 'c_2_o' in x, P.columns.values))
open_cols   = list(filter(lambda x: 'open' in x, P.columns.values))
h_2_o_cols  = list(filter(lambda x: 'h_2_o' in x, P.columns.values))
l_2_o_cols  = list(filter(lambda x: 'l_2_o' in x, P.columns.values))
vol_cols    = list(filter(lambda x: 'vol' in x, P.columns.values))

InputDF = P[open_cols].values
shape   = InputDF.shape
InputDF = np.append(InputDF,P[h_2_o_cols].values)
InputDF = np.append(InputDF,P[l_2_o_cols].values)
InputDF = np.append(InputDF,P[vol_cols].values)

InputDF  =  InputDF.reshape((4,shape[0],shape[1]))
InputDF = np.transpose(InputDF, [1,2,0])
TargetDF = P[target_cols]

input_data = InputDF
output_data = TargetDF.values

def data_i(x,length):
    res = np.zeros((x.shape[0]-length+1,length,x.shape[1],x.shape[2]))
    for i in xrange(x.shape[0]-length):
        new_data = x[i:i+length,:,:]
        res[i] = new_data
    return res

def data_o(x,length):
    res = np.zeros((x.shape[0]-length+1,length,x.shape[1]))
    for i in xrange(x.shape[0]-length):
        new_data = x[i:i+length,:]
        res[i] = new_data
    return res


#hyper parameters

output_channels = 1
kernel_size = [5,5]
lstm_high_hs = 50
lstm_low_hs = 50
hidden_size_lstm = 150
sequence_length = 30
hidden_size_ff = 317
batch_size = 50
output_size = hidden_size_ff
adam = optimizers.Adam(lr = .1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

n_days = -300
n_odays = int(n_days*(0.2))
input_data = data_i(input_data,sequence_length)
output_data = data_o(output_data,sequence_length)

input_open = input_data[n_days:,:,:,0]
input_vol = input_data[n_days:,:,:,3]
input_high = input_data[n_days:,:,:,1]
input_low = input_data[n_days:,:,:,2]
input_close= output_data[n_days:,:,:]
output_open = input_data[n_odays:,:,:,0]
output_vol = input_data[n_odays:,:,:,3]


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

vol_x = Input(shape=(sequence_length,output_size))

open_high = LSTM(lstm_high_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_high = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),name='open_high')(open_high)

open_low = LSTM(lstm_low_hs,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)

open_low = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0),name='open_low')(open_low)

data = keras.layers.Concatenate(axis=-1)([open_high,open_low,vol_x]) #remove or keep x?

data = keras.layers.core.Reshape([sequence_length,output_size,3])(data)

output_conv = Conv2D(data_format="channels_last",filters = output_channels,kernel_regularizer = keras.regularizers.l2(1.0) ,kernel_size = kernel_size,padding = "same",activation = 'relu')(data)

input_lstm = Add(axis=-1)(output_conv)

output_lstm = LSTM(hidden_size_lstm,recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(input_lstm)

output_f = Dense(hidden_size_ff,activation = 'relu',kernel_regularizer = keras.regularizers.l2(1.0))(output_lstm)

portfolio = Norm(axis=-1,name='portfolio')(output_f)

model = Model(inputs= [x,vol_x],outputs = [open_high,open_low,portfolio])
model.compile(loss= ['mean_squared_error','mean_squared_error',l_portfolio], optimizer = adam,metrics ={'portfolio': days_return},loss_weights=[1., 1.,1.])
model.fit(x=[input_open,input_vol],y=[input_high,input_low,input_close],verbose=2,batch_size=batch_size,validation_split = .2,epochs=50,callbacks=[tensorBoard])
model.save_weights('weights.h5')

