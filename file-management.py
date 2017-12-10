import glob, os

files_new = glob.glob("csv_high_new/*.csv")
files_backup = glob.glob("csv_high_new(backup)/*.csv")


def get_ticker(filepath):
  filepath = filepath[-10:-4]
  return filepath

files_new = [get_ticker(k) for k in files_new]
files_backup = [get_ticker(k) for k in files_backup]

files_remove =  set(files_new) - set(files_backup)

def remove_files(files_remove):
	for f in files_remove:
	    filepath = 'csv_high_new/' + str(f) + '.csv'
	    os.remove(filepath)

remove_files(files_remove)
# print files_remove