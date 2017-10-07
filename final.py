import glob
import pandas as pd
import numpy as np
import ntpath
import matplotlib.pyplot as plt


coloumns = np.array(['Date','Open Price','High Price','Low Price','Close Price','No. of Trades'])
files = np.array([])
all_sizes = np.array([])
all_data = pd.DataFrame()

print 'Reading data....'
files = glob.glob('/home/harshit/Desktop/Project_summer/csv/*.csv')
small_file = pd.read_csv('/home/harshit/Desktop/Project_summer/csv/539332.csv')
small_date = small_file['Date']

ret = lambda x,y: np.log(y/x) #Log return 
zscore = lambda x:(x -x.mean())/x.std() # zscore

def get_ticker(filepath):
  filepath = ntpath.basename(filepath)
  filepath = filepath[:-4]
  return filepath

def make_inputs(filepath):
    D = pd.read_csv(filepath, usecols=coloumns)
    #Load the dataframe with headers
    D =  D.loc[D['Date'].isin(small_date)]
    #D.index = pd.to_datetime(D.Date,format='%d%m%y') # Set the indix to a datetime
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



InputDF = P[input_cols].fillna(0.001)
TargetDF = P[target_cols].fillna(0.001)

def labeler(x):
    if x>0.0029:
        return 1
    if x<-0.00462:
        return -1
    else:
        return 0

def relu(x):
  if x>0:
    return x
  else:
    return 0

nprelu = np.vectorize(relu)
nplabels = np.vectorize(labeler)

print 'Reading data Done.'

class LSTM:
  
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
    """ 
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
    WLSTM[0,:] = 0 # initialize biases to zero
    if fancy_forget_bias_init != 0:
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return WLSTM
  
  @staticmethod
  def forward(X_input,Winput, Binput, Boutput, WLSTM, magic, c0 = None, h0 = None):
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """

    tanhX = np.dot(X_input,Winput)+Binput
    X = np.tanh(tanhX)
    n,b,input_size = X.shape
    d = WLSTM.shape[1]/4 # hidden size
    if c0 is None: c0 = np.zeros((b,d))
    if h0 is None: h0 = np.zeros((b,d))
    
    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
    tHout = np.zeros((n, b, d))
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content
    output = np.zeros((n,b,input_size/6))

    for t in xrange(n):
      # concat [x,h] as input to the LSTM
      prevh = Hout[t-1] if t > 0 else h0
      Hin[t,:,0] = 1 # bias
      Hin[t,:,1:input_size+1] = X[t]
      Hin[t,:,input_size+1:] = prevh
      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM)
      # non-linearities
      IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
      Ct[t] = np.tanh(C[t])
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]
      tHout[t] = np.tanh(Hout[t])
      output[t] = np.dot(tHout[t],magic[t]) + Boutput
      

    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0
    cache['output'] = output
    cache['magic'] = magic
    cache['X_input'] = X_input
    cache['X'] = X
    cache['tHout'] = tHout

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return output, C[t], Hout[t], cache
  
  @staticmethod
  def backward(doutput, cache, dcn = None, dhn = None): 

    WLSTM = cache['WLSTM']
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    output = cache['output']
    magic = cache['magic']
    X_input = cache['X_input']
    X = cache['X']
    tHout = cache['tHout']
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias
 
    # backprop the LSTM
    tanhCt = np.zeros_like(Ct)
    dBinput = np.zeros((1,input_size))
    dBoutput = np.zeros((1,input_size/6))
    dWinput = np.zeros((input_size,input_size))
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    tanhX = np.zeros(X.shape)
    dh0 = np.zeros(h0.shape)
    dc0 = np.zeros(c0.shape)
    dmagic = np.zeros(magic.shape)
    do = doutput.copy() #dloss/do (b,stock)
    dHout = np.zeros(Hout.shape)
    tanhHout = np.zeros((n,b,d))
    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(xrange(n)):
      
      #backprop through tanh
      #tanho = (1 - toutput[t]**2) * do[t]
      dBoutput += do[t].sum(0)/b
      dmagic[t] = np.dot(tHout[t].transpose(),do[t])
      
      dHout[t] += np.dot(do[t],magic[t].transpose())
      
      tanhHout[t] = (1 - tHout[t]**2) * dHout[t]
      tanhCt[t] = Ct[t]
      dIFOGf[t,:,2*d:3*d] = tanhCt[t] * tanhHout[t]
        # backprop tanh non-linearity first then continue backprop
      dC[t] += (1-tanhCt[t]**2) * (IFOGf[t,:,2*d:3*d] * tanhHout[t])

      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
      else:
        dIFOGf[t,:,d:2*d] = c0 * dC[t]
        dc0 = IFOGf[t,:,d:2*d] * dC[t]
      dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
      dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
    
    # backprop activation functions
      dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]

    # backprop matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())

    # backprop the identity transforms into Hin
      dX[t] = dHin[t,:,1:input_size+1]
      
      tanhX[t] = (1 - X[t]**2) * dX[t]
      dBinput += tanhX[t].sum(0)/b
      dWinput += np.dot(X_input[t].transpose(),tanhX[t])
      
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:]
      else:
        dh0 += dHin[t,:,input_size+1:]
 
    return dWinput, dBinput, dBoutput, dWLSTM, dc0, dh0 , dmagic


   


def main(length,batch_size,input_size,hidden_size,epochs,eta):
  n = length
  b = batch_size
  d = hidden_size
  test_n = 1
  #c0 = np.zeros((b,d))
  #h0 = np.zeros((b,d))
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)
  WLSTM = LSTM.init(input_size, d)
  Winput = np.random.randn(input_size,input_size)
  Binput = np.random.randn(1,input_size)
  Boutput = np.random.randn(1,input_size/6)
  magic = np.random.randn(n,d,input_size/6)
  #incorrect_pred = np.ones((n,b,input_size/6))
  
  X = np.zeros((n,b,input_size))
  Y = np.zeros((n,b,input_size/6)) #number of stocks are input_size/6
  test_Y = np.zeros((test_n,b,input_size/6))
  test_X = np.zeros((test_n,b,input_size))

  num_stocks = Y.shape[2]
  dWLSTM = np.zeros_like(WLSTM)
  dWinput = np.zeros_like(Winput)
  dBinput = np.zeros_like(Binput)
  dBoutput = np.zeros_like(Boutput)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dmagic = np.zeros_like(magic)

  mWLSTM = np.zeros_like(dWLSTM)
  mtWLSTM = np.zeros_like(dWLSTM)
  vWLSTM = np.zeros_like(dWLSTM)
  vtWLSTM = np.zeros_like(dWLSTM)

  mc0 = np.zeros_like(dc0)
  mtc0 = np.zeros_like(dc0)
  vc0 = np.zeros_like(dc0)
  vtc0 = np.zeros_like(dc0)

  mh0 = np.zeros_like(dh0)
  mth0 = np.zeros_like(dh0)
  vh0 = np.zeros_like(dh0)
  vth0 = np.zeros_like(dh0)
  
  mWinput = np.zeros_like(dWinput)
  mtWinput = np.zeros_like(dWinput)
  vWinput = np.zeros_like(dWinput)
  vtWinput = np.zeros_like(dWinput)
  
  mmagic = np.zeros_like(dmagic)
  mtmagic = np.zeros_like(dmagic)
  vmagic = np.zeros_like(dmagic)
  vtmagic = np.zeros_like(dmagic)


  mBinput = np.zeros_like(dBinput)
  mtBinput = np.zeros_like(dBinput)
  vBinput = np.zeros_like(dBinput)
  vtBinput = np.zeros_like(dBinput)

  mBoutput = np.zeros_like(dBoutput)
  mtBoutput = np.zeros_like(dBoutput)
  vBoutput = np.zeros_like(dBoutput)
  vtBoutput = np.zeros_like(dBoutput)
  
  for epoch in xrange(1,epochs):
    count1 = 0
    count2 = 0
    if epoch%100==0:
      eta = .9*eta
    #Creating input and output data
    for k in xrange(b):
      i = 0
      rand_k = np.random.randint(0, len(InputDF)- n - test_n - 1)
      temp_i = InputDF[rand_k : rand_k + n + test_n]
      temp_o = TargetDF[rand_k : rand_k + n + test_n]
      for j in xrange(n):
        X[j,k] = temp_i.iloc[[i]]
        Y[j,k] = temp_o.iloc[[i]]
        i+=1
      i = -test_n
      for j in xrange(test_n): 
        test_Y[j,k] = temp_o.iloc[[i]]
        test_X[j,k] = temp_i.iloc[[i]]
        i+=1


    _, _, _, cache = LSTM.forward(X, Winput, Binput, Boutput, WLSTM, magic, c0, h0)
    do = cache['output'] - Y
    
    p_output_return = 1-np.exp(cache['output'])
    p_output_labels = nplabels(p_output_return)

    p_Y_return = 1-np.exp(Y)
    p_Y_labels = nplabels(p_Y_return)

    for t in xrange(n):
      for i in xrange(b):
        for label in xrange(num_stocks):
          if p_Y_labels[t,i,label] == p_output_labels[t,i,label]:
           #incorrect_pred[t,:,label] = 0
           count1 +=1
    
    dWinput, dBinput, dBoutput, dWLSTM, dc0, dh0, dmagic = LSTM.backward(do, cache)

    _,_,_, cache_test = LSTM.forward(test_X, Winput, Binput, Boutput, WLSTM, magic, c0, h0)

    
    test_output_return = 1-np.exp(cache_test['output'])
    test_output_labels = nplabels(test_output_return)


    test_Y_return = 1-np.exp(test_Y)
    test_Y_labels = nplabels(test_Y_return)    
    for i in xrange(b):
      for label in xrange(num_stocks):
        if test_Y_labels[0,i,label] == test_output_labels[0,i,label]:
          count2 +=1
      

    #update all params
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-7

    #adam update
    
    mWLSTM = beta1*mWLSTM + (1-beta1)*dWLSTM
    mtWLSTM = mWLSTM / (1-beta1**epoch)
    vWLSTM = beta2*vWLSTM + (1-beta2)*(dWLSTM**2)
    vtWLSTM = vWLSTM / (1-beta2**epoch)
    WLSTM += - eta * mtWLSTM / (np.sqrt(vtWLSTM) + eps)

    

    mc0 = beta1*mc0 + (1-beta1)*dc0
    mtc0 = mc0 / (1-beta1**epoch)
    vc0 = beta2*vc0 + (1-beta2)*(dc0**2)
    vtc0 = vc0 / (1-beta2**epoch)
    c0 += - eta * mtc0 / (np.sqrt(vtc0) + eps)

    mh0 = beta1*mh0 + (1-beta1)*dh0
    mth0 = mh0 / (1-beta1**epoch)
    vh0 = beta2*vh0 + (1-beta2)*(dh0**2)
    vth0 = vh0 / (1-beta2**epoch)
    h0 += - eta * mth0 / (np.sqrt(vth0) + eps)

    mWinput = beta1*mWinput + (1-beta1)*dWinput
    mtWinput = mWinput / (1-beta1**epoch)
    vWinput = beta2*vWinput + (1-beta2)*(dWinput**2)
    vtWinput = vWinput / (1-beta2**epoch)
    Winput += - eta * mtWinput / (np.sqrt(vtWinput) + eps)
    
    mmagic = beta1*mmagic + (1-beta1)*dmagic
    mtmagic = mmagic / (1-beta1**epoch)
    vmagic = beta2*vmagic + (1-beta2)*(dmagic**2)
    vtmagic = vmagic / (1-beta2**epoch)
    magic += - eta * mtmagic / (np.sqrt(vtmagic) + eps)

    mBinput = beta1*mBinput + (1-beta1)*dBinput
    mtBinput = mBinput / (1-beta1**epoch)
    vBinput = beta2*vBinput + (1-beta2)*(dBinput**2)
    vtBinput = vBinput / (1-beta2**epoch)
    Binput += - eta * mtBinput / (np.sqrt(vtBinput) + eps)
    
    mBoutput = beta1*mBoutput + (1-beta1)*dBoutput
    mtBoutput = mBoutput / (1-beta1**epoch)
    vBoutput = beta2*vBoutput + (1-beta2)*(dBoutput**2)
    vtBoutput = vBoutput / (1-beta2**epoch)
    Boutput += - eta * mtBoutput / (np.sqrt(vtBoutput) + eps)


    """ Simple sgd update
    Winput = Winput - eta*dWinput
    Binput = Binput - eta*dBinput
    WLSTM = WLSTM - eta*dWLSTM
    c0 = c0 - eta*dc0
    h0 = h0 - eta*dh0
    magic = magic - eta*dmagic
    """
    """
    unique, counts = np.unique(test_output_labels, return_counts=True)
    print dict(zip(unique, counts))
    unique2, counts2 = np.unique(test_Y_labels, return_counts=True)
    print dict(zip(unique2, counts2))
    """

    print np.sum(np.absolute(dWLSTM))
    print np.sum(np.absolute(dc0))
    print np.sum(np.absolute(dh0))
    print np.sum(np.absolute(dmagic))
    print np.sum(np.absolute(dWinput))
    print np.sum(np.absolute(dBinput))
    print np.sum(np.absolute(dBoutput))
    
    print count1/b
    print count2/b
  
    loss_temp = np.sum(np.absolute(do)) 
    loss_2 = loss_temp
    print loss_2

    print "Epoch {0} complete".format(epoch)
    



main(5,60,1788,300,10000,.001)


def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n,b,d = (5, 2, 4) # sequence length, batch size, hidden size
  input_size = 12
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)
  Winput = np.random.randn(input_size,input_size)
  Binput = np.random.randn(1,input_size)
  Boutput = np.random.randn(1,input_size/6)
  Y = np.zeros((n,b,input_size/6))
  magic = np.random.randn(n,d,input_size/6)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, Winput, Binput, Boutput, WLSTM, magic, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dWinput, dBinput, dBoutput, dWLSTM, dc0, dh0, dmagic = LSTM.backward(dH, cache)
  def fwd():
    h,_,_,_ = LSTM.forward(X, Winput, Binput, Boutput, WLSTM, magic, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-7
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [Winput,Binput, Boutput, WLSTM, c0, h0, magic]
  grads_analytic = [dWinput,dBinput, dBoutput, dWLSTM, dc0, dh0, dmagic]
  names = ['Winput','Binput', 'Boutput', 'WLSTM', 'c0', 'h0', 'magic']
  for j in xrange(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in xrange(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0 # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0 # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)

#checkBatchGradient()
#if __name__ == "__main__":

#  checkSequentialMatchesBatch()
#  raw_input('check OK, press key to continue to gradient check')
#  checkBatchGradient()
#print 'every line should start with OK. Have a nice day!'


