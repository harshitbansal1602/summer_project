import glob
import pandas as pd

files1=[]
new_files=[]
files = glob.glob('csv_high_new(backup)/*.csv')
for f in files:
	f = f[-10:-4]
	f = int(f)
	new_files.append(f)

# data = pd.read_csv("csv_high_new.csv")
files = glob.glob('csv_high_new/*.csv')
for f in files:
	f = f[-10:-4]
	f = int(f)
	files1.append(f)

# companies = data['Security Code']

print  set(new_files) - set(files1)

for f in files:
	data = pd.read_csv(f)
	if data['Date'][2] != '2-November-2017':
		print f

