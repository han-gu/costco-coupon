import tensorflow as tf 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import math
import nltk
from datetime import datetime, timedelta
import calendar

seed = 128

# number of neurons in each layer
input_num_units = 252
hidden_num_units = 200
output_num_units = 1

x = tf.placeholder(tf.float32, [None, input_num_units])


weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}
#set up layers
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

df=pd.read_excel('products.xlsx')
keywords = df['products'].values

print("Type in a product please.")
query = input()

queryWords = nltk.word_tokenize(query)

inputVector = [None]*len(keywords)
for word in queryWords:
	for i in range(0,len(keywords)):
		if keywords[i] in word:
			inputVector[i]=1
		else:
			inputVector[i]=0
inputVector = numpy.float32(inputVector).reshape(1,252)

saver = tf.train.Saver()

with tf.Session() as session:
	saver.restore(session, "model")
	#feed vector into model
	feed_dict = {x: inputVector}
	daysUntilSale = round(numpy.asscalar((session.run(output_layer, feed_dict))))
	#Output predicted value
	monthOfSale = (datetime.now() + timedelta(days=daysUntilSale)).month
	print(query + " will go on sale in the month of " + calendar.month_name[monthOfSale])
