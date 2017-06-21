#all data collected june 12

import urllib.request
import re
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime, timedelta

b = open('data-raw.csv', 'w')
a = csv.writer(b)

currentDate=datetime.now()

myBaseUrl = "http://www.adamcos.com/archive.php"
with urllib.request.urlopen(myBaseUrl) as baseUrl:
	baseRawHTML=baseUrl.read()
baseSoup = BeautifulSoup(baseRawHTML, 'html.parser')

extensionUrls=[]
for linkElement in baseSoup.findAll('a',href=True):
	extensionUrls.append(linkElement['href'])

otherDiscounts = []
totalDataRow=[]

#change accordingly for data points without dollars off or comma delimited which starts at March 3, 2010
for k in range(0,1):
	extension=extensionUrls[k][7:]
	#important
	month = extension[4:6]
	prevDate = datetime(year=int(extension[0:4]), month=int(extension[4:6]), day=int(extension[6:8]))
	#important
	daysSinceSale=abs((currentDate-prevDate).days)
	print(extension)

	myUrl = "http://www.adamcos.com/Costco/" + extension
	with urllib.request.urlopen(myUrl) as url:
		rawHTML=url.read()
	soup = BeautifulSoup(rawHTML, 'html.parser')

	#Get the number of pages in the coupon book
	#Figure out how to get over stupid insert
	
	paragraphElements = soup.findAll('p')
	allH3 = soup.findAll('h3')
	allH3WithPageNumber = [member for member in allH3 if "age" in member.text]
	lastPageElement = allH3[len(allH3WithPageNumber)-1]
	lastPage = lastPageElement.text[5:]

	#Get all discounts in coupon book
	allPriceDiscounts = []

	for i in range (0,int(lastPage)):
		try:
			allPriceDiscounts.extend(paragraphElements[i].text.split(';'))
		except:
			pass
	#Get sale and item name 
	#important
	for j in range(0,len(allPriceDiscounts)):
		try:
			dataRow=[]
			#sale
			dataRow.append(allPriceDiscounts[j][:allPriceDiscounts[j].index('off')])
			#item
			dataRow.append(allPriceDiscounts[j][allPriceDiscounts[j].index('off')+4:])
			dataRow.append(daysSinceSale)
			dataRow.append(month)
			totalDataRow.append(dataRow)
		except:
			otherDiscounts.append(allPriceDiscounts[j])

a.writerows(totalDataRow)
b.close()

###Clean the data

b = open('data-clean.csv', 'w')
a = csv.writer(b)

totaldata=[]

file = open("data-raw.csv","r")
for line in file:
	line = re.sub('[",]', '', line)
	print(line)
	totaldata.append(line.split("\t"))

a.writerows(totaldata)
b.close()

