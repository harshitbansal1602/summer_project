import glob
import pandas as pd
import numpy as np

ret = lambda x,y: np.log(x/y) #Log return
zscore = lambda x:(x - min(x))/(max(x) - min(x)) # zscore


def get_ticker(filepath):
  filepath = filepath[-10:-4]
  return filepath

def make_inputs(filepath,columns,small_date):
    D = pd.read_csv(filepath, usecols=columns)
    D = D.drop_duplicates(subset = ['Date'])
    D =  D.loc[D['Date'].isin(small_date)]
    D = D.reset_index(drop=True)
    # D = D[:400] # to get only starting 400 values in the common dates
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)

    Res['o1_o0'] = ret(D['Open Price'],D['Open Price'].shift(-1)).fillna(0)
    Res['close'] = D['Close Price']
    Res['open'] = D['Open Price']
    Res['return'] = ret(D['Close Price'],D['Close Price'].shift(-1)).fillna(0) #todays return
    Res['vol'] = D['No.of Shares'].shift(-1).fillna(0)
    Res['ticker'] = ticker
    return Res

def companies():
    files = glob.glob("/home/harshit/Desktop/Project_summer/csv_high_new/*.csv")
    companies = [get_ticker(file) for file in files]
    companies.sort()
    return companies

def read_data():
    print 'read'
    # columns = np.array(['Date','Open Price','High Price','Low Price','Close Price','No.of Shares'])
    columns = np.array(['Date','Open Price','Close Price','No.of Shares'])
    files = np.array([])
    all_sizes = np.array([])
    all_data = pd.DataFrame()

    files = glob.glob("/home/harshit/Desktop/Project_summer/csv_high_new/*.csv")
    small_date = pd.read_csv(files[0], usecols=['Date']).values
    for f in files:
        date = pd.read_csv(f, usecols=['Date']).values
        small_date = np.intersect1d(small_date, date)
     
    for f in files:
      Res = make_inputs(f,columns,small_date)
      all_data = all_data.append(Res)

    pivot_columns = all_data.columns[:-1]
    P = all_data.pivot_table(index=all_data.index,columns='ticker',values=pivot_columns)
    mi = P.columns.tolist()
    new_ind = pd.Index(e[1] +'_' + e[0] for e in mi)
    P.columns = new_ind
    P = P.sort_index(axis=1,ascending=True) # Sort by columns
    P = P.reindex(index=P.index[::-1]) #latest at last
    return P

