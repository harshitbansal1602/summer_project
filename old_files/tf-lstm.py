import tensorflow as tf
import glob
import pandas as pd
import numpy as np


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
print len(small_date)
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
input_cols  = list(filter(lambda x: 'c1_c0' not in x, P.columns.values))

InputDF = P[input_cols]
TargetDF = P[target_cols]

#I currently have 582 days of data so making approx 9:1 ratio of train:test data

train_input_data = InputDF.loc[:523]
train_output_data = TargetDF.loc[:523]
test_input_data = InputDF.loc[523:]
test_output_data = TargetDF.loc[523:]


def labeler(x):
    if tf.cond(x>0.0029):
        return 1.0
    if x<-0.00462:
        return -1.0
    else:
        return 0.0


nplabels = np.vectorize(labeler)

sess = tf.InteractiveSession()

def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
	initial = tf.constant(0.1, shape=shape) #initially set to 0.1 as relu activation is used
  	return tf.Variable(initial)


def model(x, sequence_length, batch_size,hidden_size_f1, hidden_size_ff, hidden_size_lstm):
	#too-do make the ff on top of the rnn cell and make sequence

	lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm,forget_bias=3)	

	initial_state = lstm_cell.zero_state(batch_size,tf.float32)

	lstm_input = tf.layers.dense(x,hidden_size_f1,activation = tf.nn.relu,use_bias = True,kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),kernel_regularizer = tf.contrib.layers.l2_regularizer)

	lstm_layer = tf.nn.dynamic_rnn(lstm_cell,lstm_input,initial_state= initial_state)

	output_lstm,state = lstm_layer

	output_f = tf.layers.dense(output_lstm,hidden_size_ff,activation = tf.nn.relu,use_bias = True,kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),kernel_regularizer = tf.contrib.layers.l2_regularizer)

	return output_f,state


def main():
	
	#hyperparameters
	epochs = 100000 #number of epochs you want to run
	batch_size = 350 #Batch size
	sequence_length = 15 #number of contineous words/days as input
	hidden_size_lstm = 300 #hidden units in lstm cell
	hidden_size_ff = 295 # hidden size of last ff connected layer.Note- this will the output size too.
	hidden_size_f1 = 400 # hidden size of first ff connected layer.
	input_size = 1770 # size of the word embedding or similar 
	learning_rate = 1e-2
	#starter_learning_rate = 1.0 #starting-learning rate for the model

	output_size = hidden_size_ff #output size will be equal to hidden size of last layer

	
	x = tf.placeholder(tf.float32,shape=((batch_size, sequence_length, input_size)),name='input')
	y = tf.placeholder(tf.float32,shape=((batch_size, sequence_length, output_size)),name='output')
	y_pred , state = model(x,sequence_length, batch_size, hidden_size_f1, hidden_size_ff, hidden_size_lstm) #feed-forward step

	pred_return = tf.exp(y_pred) - 1 # = (y-x)/x , profit/loss

	print pred_return.shape
	output_return = y # actual profit/loss
	
	loss = tf.losses.mean_squared_error(labels = output_return, predictions = pred_return, reduction = tf.losses.Reduction.MEAN)
	
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) #optimiser step (deafult = 'adam_optimiser')
	
	#Initializing the i/o
	X = np.zeros((batch_size, sequence_length, input_size)) 
	Y = np.zeros((batch_size, sequence_length, output_size))
	test_X = np.zeros((batch_size, sequence_length, input_size)) 
	test_Y = np.zeros((batch_size, sequence_length, output_size))  

	tf.global_variables_initializer().run() #initializing all variables
	writer = tf.summary.FileWriter('/home/harshit/Desktop/Project_summer/graph',sess.graph)	#summary writer dic = 'desired/directory'

	for epoch in xrange(epochs):
		for k in xrange(batch_size):  
			#making batches
			train_k = np.random.randint(0, len(train_input_data)- sequence_length - 1)
			test_k = np.random.randint(0, len(test_input_data)- sequence_length - 1)

			train_i = train_input_data[train_k : train_k + sequence_length]
			train_o = train_output_data[train_k : train_k + sequence_length]
			
			test_i = test_input_data[test_k : test_k + sequence_length]
			test_o = test_output_data[test_k : test_k + sequence_length]
			
			for j in xrange(sequence_length):
				X[k,j] = train_i.iloc[[j]]
				Y[k,j] = train_o.iloc[[j]]
				test_Y[k,j] = test_o.iloc[[j]]
				test_X[k,j] = test_i.iloc[[j]]

		if epoch%5 == 0:
			
			learning_rate = .99*learning_rate
			predictions = sess.run(pred_return,feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
			test_predictions = sess.run(pred_return,feed_dict={x: test_X.astype(np.float32), y: test_Y.astype(np.float32)})
			
			pred_labels = nplabels( np.exp(predictions[:,-1,:]) - 1)
			test_labels = nplabels(np.exp(test_predictions[:,-1,:]) - 1)
			
			return_labels = nplabels( np.exp(Y[:,-1,:]) - 1)
			return_labels_test = nplabels( np.exp(test_Y[:,-1,:]) - 1)

			count = 0.0
			count1 = 0.0
			for i in xrange(batch_size):
				for label in xrange(hidden_size_ff):
					if pred_labels[i,label] == return_labels[i,label]:
						count +=1.0
					if test_labels[i,label] == return_labels_test[i,label]:
						count1 += 1.0
			unique, counts = np.unique(pred_labels, return_counts=True)
			print dict(zip(unique, counts))
			unique, counts = np.unique(return_labels, return_counts=True)
			print dict(zip(unique, counts))
			
			print count/(batch_size*hidden_size_ff)
			print loss.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
			
			unique2, counts2 = np.unique(test_labels, return_counts=True)
			print dict(zip(unique2, counts2))
			unique2, counts2 = np.unique(return_labels_test, return_counts=True)
			print dict(zip(unique2, counts2))
			
			print count1/(batch_size*hidden_size_ff)
			#print correct_pred.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})/(output_size*batch_size*sequence_length)
			
		#print unique.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)}) + '  ' + counts2.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
		#print unique2.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)}) + '  ' + counts.eval(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
		train_step.run(feed_dict={x: X.astype(np.float32), y: Y.astype(np.float32)})
		print 'Epoch ' + str(epoch) + ' completed.'
		
	writer.close()

main()


