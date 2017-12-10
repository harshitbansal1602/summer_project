from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dateutil.parser import parse

import pandas as pd
import time
import os
import datetime
import glob

now = datetime.datetime.now()

files = glob.glob("csv_high_new/*.csv")
def get_ticker(filepath):
  filepath = filepath[-10:-4]
  return filepath
# create a new Firefox session
companies = [get_ticker(file) for file  in files] 
driver = webdriver.Firefox()
driver.implicitly_wait(5)


def update(company):

	driver.get("http://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx?expandable=7&flag=0")
	search_field = driver.find_element_by_id("ctl00_ContentPlaceHolder1_GetQuote1_smartSearch")
	date_initial_field = driver.find_element_by_id("ctl00_ContentPlaceHolder1_txtFromDate")
	date_final_field = driver.find_element_by_id("ctl00_ContentPlaceHolder1_txtToDate")
	submit = driver.find_element_by_id("ctl00_ContentPlaceHolder1_btnSubmit")
	search_field.clear()
	date_final_field.clear()
	date_initial_field.clear()
	for char in str(company):
		search_field.send_keys(char)
		time.sleep(.3)

	list_item = driver.find_elements_by_css_selector('#listEQ>li:first-child')
	
	if list_item[0].text == "No Match found":
		return 0
	else :
		list_item[0].click()
		date_initial_field.send_keys("24/10/2017")
		driver.execute_script("document.getElementById('ctl00_ContentPlaceHolder1_txtToDate').setAttribute('onkeypress','');")
		date_final_field.send_keys(now.strftime("%d/%m/%Y"))
		submit.click()
		element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_btnDownload1")))
		source = driver.page_source
		soup = BeautifulSoup(source)
		span = soup.find('span', {'id' : 'ctl00_ContentPlaceHolder1_spnStkData'})
		table = span.find('table')
		df = pd.read_html(str(table))[0]
		return df.iloc[-1]

# navigate to the application home page
# enter search keyword and submit

for company in companies:
	count = 0
	file_path = "csv_high_new/" + str(company) + ".csv"
	df = update(company)
	new = df.values
	new[0] = parse(new[0]).strftime('%d-%B-%Y')
	old = pd.read_csv(file_path)
	with open(file_path, 'w') as f:
		old.loc[-1] = new  # adding a row
		old.index = old.index + 1  # shifting index
		old = old.sort_index()  # sorting by index
		old.to_csv(f)
	print company
