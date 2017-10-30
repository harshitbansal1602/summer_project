import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import keras
from keras.layers import LSTM, Conv2D, Dense, Activation, Add, Flatten, TimeDistributed
from keras.models import Model
import keras.backend as K

columns = np.array(['Date','Open Price','High Price','Low Price','Close Price','No. of Trades'])
files = np.array([])
all_sizes = np.array([])
all_data = pd.DataFrame()

print 'Reading data....'
files = glob.glob("csv_high/*.csv")
small_date = pd.read_csv(files[0], usecols=['Date']).values
for f in files:
    date = pd.read_csv(f, usecols=['Date']).values
    small_date = np.intersect1d(small_date, date)  
    #print str(len(small_date)) + ' ' + small_date[0] + ' ' + str(f[-10:-4])
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

    Res['c_2_o'] = ret(D['Open Price'],D['Close Price'])
    Res['h_2_o'] = ret(D['Open Price'],D['High Price'])
    Res['l_2_o'] = ret(D['Open Price'],D['Low Price'])
    Res['c_2_h'] = ret(D['Close Price'],D['High Price'])
    Res['h_2_l'] = ret(D['High Price'],D['Low Price'])
    Res['c1_c0'] = ret(D['Close Price'],D['Close Price'].shift(-1)).fillna(0) #todays return
    Res['vol'] = zscore(D['No. of Trades'])
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
c_2_o_cols  = list(filter(lambda x: 'c_2_o' in x, P.columns.values))
h_2_o_cols  = list(filter(lambda x: 'h_2_o' in x, P.columns.values))
l_2_o_cols  = list(filter(lambda x: 'l_2_o' in x, P.columns.values))
c_2_h_cols  = list(filter(lambda x: 'c_2_h' in x, P.columns.values))
h_2_l_cols  = list(filter(lambda x: 'h_2_l' in x, P.columns.values))
vol_cols    = list(filter(lambda x: 'vol' in x, P.columns.values))

InputDF = P[c_2_o_cols].values
shape   = InputDF.shape
InputDF = np.append(InputDF,P[h_2_o_cols].values)
InputDF = np.append(InputDF,P[l_2_o_cols].values)
InputDF = np.append(InputDF,P[c_2_h_cols].values)
InputDF = np.append(InputDF,P[h_2_l_cols].values)
InputDF = np.append(InputDF,P[vol_cols].values)

InputDF  =  InputDF.reshape((6,shape[0],shape[1]))
InputDF = np.transpose(InputDF, [1,2,0])
TargetDF = P[target_cols]

train_input_data = InputDF[:523,:,:]
train_output_data = TargetDF.loc[:522]
test_input_data = InputDF[523:,:,:]
test_output_data = TargetDF.loc[523:]


def labeler(x):
    if x>0.0029:
        return 1
    if x<-0.00462:
        return -1
    else:
        return 0


nplabels = np.vectorize(labeler)

sess = tf.InteractiveSession()


def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape) #initially set to 0.1 as relu activation is used
    return tf.Variable(initial)

def model(x):

    #  SHAPE(X) = (BATCH_SIZE, H, W, NUM_CHANNELS)
    #  SHAPE(Y) = (BATCH_SIZE, H, W)

    x = Conv2D(num_filters=2, kernel_size=(K_H, K_W), strides=(1,1), padding=PADDING)(x)
    x = Activation('relu')(x)
    x = Add()(x)
    print x.shape

    x = LSTM(HIDDEN_SIZE, activation='tanh', recurrent_dropout=0.2, dropout=0.5, return_sequences=True)(x)
    x = TimeDistributed(Dense(W))(x)


   return x

def model(x, sequence_length, batch_size, hidden_size_ff, hidden_size_lstm):
    
    output_channels = 2
   
    output_conv = tf.layers.conv2d(x,filters = output_channels,kernel_size = [5,5],padding = "same", data_format = "channels_last",activation = tf.nn.elu )
    
    input_lstm = tf.reduce_sum(output_conv,axis=3)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm,forget_bias=3)   
    initial_state = lstm_cell.zero_state(batch_size,tf.float32)

    lstm_layer = tf.nn.dynamic_rnn(lstm_cell,input_lstm,initial_state= initial_state)

    output_lstm,state = lstm_layer
    
    output_f = tf.layers.dense(output_lstm,hidden_size_ff,activation = tf.nn.elu ,use_bias = True,kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),kernel_regularizer = tf.contrib.layers.l2_regularizer)

    return output_f,state


def main():
    
    #hyperparameters
    epochs = 100000 #number of epochs you want to run
    batch_size = 350 #Batch size
    sequence_length = 14 #number of contineous words/days as input
    hidden_size_lstm = 300 #hidden units in lstm cell
    hidden_size_ff = 295 # hidden size of last ff connected layer.Note- this will the output size too.
    input_size = 1770 # size of the word embedding or similar 
    learning_rate = 1e-2 
    output_size = hidden_size_ff #output size will be equal to hidden size of last layer

    x = tf.placeholder(tf.float32,shape=((batch_size, sequence_length, output_size, 6)))

    y = tf.placeholder(tf.float32,shape=((batch_size,sequence_length,output_size)),name='output')
    
    y_pred , state = model(x,sequence_length,batch_size,hidden_size_ff,hidden_size_lstm) #feed-forward step
    
    pred_return = tf.exp(y_pred) - 1 # = (y-x)/x , profit/loss
    
    output_return = tf.exp(y) - 1 # actual profit/loss
    loss = tf.losses.mean_squared_error(labels = output_return, predictions = pred_return, reduction = tf.losses.Reduction.MEAN)
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) #optimiser step 
    
    #Initializing the i/o
    X = np.zeros((batch_size, sequence_length, output_size, 6)) 
    Y = np.zeros((batch_size,sequence_length,output_size))
    test_X = np.zeros((batch_size, sequence_length, output_size, 6)) 
    test_Y = np.zeros((batch_size,sequence_length,output_size))  

    tf.global_variables_initializer().run() #initializing all variables
    writer = tf.summary.FileWriter('/home/harshit/Desktop/Project_summer/graph',sess.graph) #summary writer dic = 'desired/directory'

    for epoch in xrange(epochs):
        
        for k in xrange(batch_size):  
            #making batches
            train_k = np.random.randint(0, train_input_data.shape[0]- sequence_length - 1)
            test_k = np.random.randint(0, test_input_data.shape[0]- sequence_length - 1)

            X[k] = train_input_data[train_k : train_k + sequence_length,:,:]
            train_o = train_output_data[train_k : train_k + sequence_length]
            
            test_X[k] = test_input_data[test_k : test_k + sequence_length,:,:]
            test_o = test_output_data[test_k : test_k + sequence_length]
            
            for j in xrange(sequence_length):
                Y[k,j] = train_o.iloc[[j]]
                test_Y[k,j] = test_o.iloc[[j]]

        if epoch%5 == 0:
            
            learning_rate = .99*learning_rate
            predictions = sess.run(pred_return,feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
            test_predictions = sess.run(pred_return,feed_dict={x: test_X.astype(np.float32), y: test_Y.astype(np.float32)})
            pred_labels = nplabels(predictions[:,-1,:])
            test_labels = nplabels(test_predictions[:,-1,:])
         
            return_labels = nplabels(np.exp(Y[:,-1,:]) - 1)
            return_labels_test = nplabels(np.exp(test_Y[:,-1,:]) - 1)
            
            count = 0.0
            count1 = 0.0
            for i in xrange(batch_size):
                for label in xrange(hidden_size_ff):
                    if pred_labels[i,label] == return_labels[i,label]:
                        count +=1.0
                    if test_labels[i,label] == return_labels_test[i,label]:
                        count1 += 1.0
            
            print count/(batch_size*hidden_size_ff)
            print loss.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
            print count1/(batch_size*hidden_size_ff)
        
        train_step.run(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
        print 'Epoch ' + str(epoch) + ' completed.'
        
    writer.close()

main()

