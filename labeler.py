#Prepare data and translate from words to vectors.

import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import math
import pickle
import editdistance
import nltk
import csv

#read in vector data
#for every column if there's a one then add the item to a dictionary of word:list
#within each word in the dictionary, calculate the days since last sale for each item


file = pd.read_excel("finaldata.xlsx")
items = file['item'].values.tolist()

with open('totalVectorData.p', 'rb') as fp:
    totalVectorData = pickle.load(fp)

daysSince = file['dayssince'].values.tolist()

keywords = totalVectorData[0]
totalVectorData = totalVectorData[1:]




categoriesByKeyword=[None]*len(keywords)

"""
categoriesByKeyword[0].append(1)
categoriesByKeyword[0].append(1)

print(categoriesByKeyword[1])


for j in range(0,len(keywords)):
	if totalVectorData[0][j]  == 1:
		print(totalVectorData[0])
		if categoriesByKeyword[j]
		categoriesByKeyword[j].append(items[0])
"""
#already saved misc data, ie. the data that didn't match any of the keywords. misc.p isn't updated
#need to save days since data
for i in range(0,len(totalVectorData)):
	for j in range(0,len(keywords)):
		if totalVectorData[i][j]  == 1:
			if categoriesByKeyword[j] == None:
				addedItem = [str(items[i]) + "#" + str(daysSince[i])]
				categoriesByKeyword[j]=addedItem
			else:
				categoriesByKeyword[j].append(str(items[i]) + "#" + str(daysSince[i]))



for listElement in categoriesByKeyword:
	total=0
	count=0
	for i in range(0,len(listElement)):
		itemName = listElement[i][:listElement[i].index("#")]
		daysSince = listElement[i][listElement[i].index("#")+1:]
		found=False
		for j in range(i,len(listElement)):
			if found == False:
				nextItemName = listElement[j][:listElement[j].index("#")]
				nextDaysSince = listElement[j][listElement[j].index("#")+1:]
				if nextDaysSince != daysSince:
					total+=int(daysSince) - int(nextDaysSince)
					count+=1
					listElement[i]=itemName + "#" + str(int(daysSince) - int(nextDaysSince))
					found=True
		#if the item wasn't discounted again in this particular keyword category
		if found == False:
			if count!=0:
				listElement[i]=itemName+"#"+str(total/count)
			else:
				listElement[i]=itemName+"#"+"***"


#USE THE AVERAGE OF THE OTHER DATAPOINTS IN ITS CATEGORY
#MAKE AN EXCEL DOC WITH THE LABELS CORRESPONDING TO THE RIGHT ITEMS
finalDict=dict()

for listElement in categoriesByKeyword:
	for i in range(0,len(listElement)):
		itemName = listElement[i][:listElement[i].index("#")]
		label = listElement[i][listElement[i].index("#")+1:]
		if itemName in finalDict:
			finalDict[itemName]=min(finalDict[itemName],label)
		else:
			finalDict[itemName]=label

finalItems=list(finalDict.keys())
finalLabels=list(finalDict.values())

b = open('AlphaDeta.csv', 'w')
a = csv.writer(b)

totalData=[]

for i in range(0,len(finalItems)):
	dataRow=[]
	dataRow.append(finalItems[i])
	dataRow.append(finalLabels[i])
	totalData.append(dataRow)

a.writerows(totalData)
b.close()

print(keywords[0])

#Going to combine these two 
d = open('BetaDeta.csv', 'w')
c = csv.writer(d)

totalTotalData=[]
for item in finalItems:
	print(item)
	dataDataRow=[None]*len(keywords)
	for i in range(0,len(keywords)):
		if keywords[i] in item:
			dataDataRow[i]=1
		else:
			dataDataRow[i]=0
	totalTotalData.append(dataDataRow)

c.writerows(totalTotalData)
d.close()


