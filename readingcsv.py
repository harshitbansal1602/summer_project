from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import time
import os

data = pd.read_csv("/home/harshit/Desktop/Project_summer/companies.csv")
companies = data['Security Code']
print companies
# create a new Firefox session
driver = webdriver.Firefox()
driver.implicitly_wait(5)


def download(company):
	
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
	list_item = []
	if driver.find_elements_by_css_selector('#listEQ>li:first-child'):
		list_item = driver.find_elements_by_css_selector('#listEQ>li:first-child')
		list_item[0].click()
	else :
		list_item = driver.find_elements_by_css_selector('#listtest>li:first-child')
		list_item[0].text == "No Match found"
		return 1	
	date_initial_field.send_keys("01/01/2000")
	driver.execute_script("document.getElementById('ctl00_ContentPlaceHolder1_txtToDate').setAttribute('onkeypress','');")
	date_final_field.send_keys("01/01/2017")
	submit.click()
	element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_btnDownload1")))
	downlaod_pic = driver.find_element_by_id("ctl00_ContentPlaceHolder1_btnDownload1")
	downlaod_pic.click()
	return 0
# navigate to the application home page
# enter search keyword and submit


for company in companies:
	count = 0
	file_path = "/home/harshit/Downloads/" + str(company) + ".csv"
	while not os.path.exists(file_path):
		if download(company) == 1:
			print 'not found'
			i = data.set_index('Security Code').index.get_loc(company)
			data.drop(df.index[[i]], inplace=True)
			data.to_csv('ucompanies.csv')
			break
		while not os.path.exists(file_path):
			if count < 5:
				count += 1
				time.sleep(1)
			else:
				break		
 		print company

# for company in companies:
# 	count = 0
# 	file_path = "/home/harshit/Downloads/" + str(company) + ".csv"
# 	while not os.path.exists(file_path):
# 		download(company)
# 		while not os.path.exists(file_path):
# 			if count < 5:
# 				count += 1
# 				time.sleep(1)
# 			else :
# 				download(company)
# 		print company
