import tensorflow as tf 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import math
import nltk

df=pd.read_excel('AlphaData.xlsx',header=None)

totalLabels = df[0]
totalData = df.ix[:,1:]

trainingData=totalData[:4000]
testingData=totalData[4000:]
trainingLabels=totalLabels[:4000]
testingLabels=totalLabels[4000:]

trainingDataNumpy = numpy.float32(trainingData)
trainingLabelsNumpy = numpy.float32(trainingLabels)
testingDataNumpy = numpy.float32(testingData)
testingLabelsNumpy = numpy.float32(testingLabels)


#Setting up
seed = 128
rng = numpy.random.RandomState(seed)

# number of neurons in each layer
input_num_units = len(totalData.columns)
hidden_num_units = 200
output_num_units = 1

def preproc(unclean_batch_x):
    #Convert values to range 0-1
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    #Create batch with random samples and return appropriate format
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('trainingDataNumpy')[[batch_mask]].reshape(-1, input_num_units)
    
    batch_y = eval('trainingLabelsNumpy')[[batch_mask]].reshape(-1,output_num_units)
        
    return batch_x, batch_y




x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 600
batch_size = 128
learning_rate = 0.001

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

#define cost
cost = tf.reduce_mean(tf.squared_difference(output_layer, y))

#adam gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initialize variables
init=tf.global_variables_initializer()

#Takes user input and makes inputVector for model
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

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(trainingDataNumpy.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, trainingDataNumpy.shape[0],'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print("\nTraining complete!")
    
    
    # rewrite prediction on testing data
    pred_temp = tf.less_equal(tf.abs(tf.subtract(tf.round(output_layer), tf.round(y))),30)
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:", accuracy.eval({x: testingDataNumpy.reshape(-1, input_num_units), y: testingLabelsNumpy.reshape(-1,output_num_units)}))
    saver.save(sess, "model")


with tf.Session() as session:
	saver.restore(session, "model")
	#feed vector into model
	feed_dict = {x: inputVector}
	classification = session.run(output_layer, feed_dict)
	#Output predicted value
	print(classification)












