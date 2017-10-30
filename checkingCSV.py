import glob
import pandas as pd

files=[]
new_files=[]
files = glob.glob('/home/harshit/Downloads/csv/*.csv')
for f in files:
	f = f[-10:-4]
	f = int(f)
	new_files.append(f)

data = pd.read_csv("/home/harshit/Desktop/Project_summer/companies_high.csv")
companies = data['Security Code']

print set(new_files) - set(companies.values) 

