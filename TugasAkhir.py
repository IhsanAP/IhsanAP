import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

# Input Data saham
df = pd.read_csv('TLKM.JK.csv')
df.drop(['Open','High','Low','Close','Volume'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index (['Date'], drop=True)

# Setting jumlah series sebanyak 20 series (data training)
DF = np.array(df)
num_periods = 20
f_horizon = 1
x_data = DF[:(len(DF) - (len(DF) % num_periods))]
x_batches = x_data.reshape(-1,20,1)
y_data = DF[1:(len(DF)-(len(DF) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)
print (len(x_batches))
print (x_batches.shape)
print (x_batches[0:2])
print (y_batches[0:1])
print (y_batches.shape)

# Create data test 
def test_data(series,forecast,num_periods):
    test_x_setup = DF[-(num_periods+forecast):]
    testX = test_x_setup[:num_periods].reshape(-1,20,1)
    testY = DF[-(num_periods):].reshape(-1,20,1)
    return testX, testY
X_test,Y_test = test_data(DF,f_horizon,num_periods)
print (X_test.shape)
print (X_test)

# Implementasi tensorflow untuk model prediksi menggunakan model prediksi Relu, 
# variable hidden 100, inputs 1 (data historis saham), output 1 (prediksi harga saham), 
# model menggunakan reccurent neural network
tf.reset_default_graph()
num_periods = 20
inputs = 1
hidden = 100 
output = 1
X = tf.placeholder(tf.float32,[None, num_periods, inputs])
y = tf.placeholder(tf.float32,[None, num_periods, output])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation = tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
learning_rate = 0.001 #Stochastic Gradient Descent rate : 0.001
stacked_rnn_output = tf.reshape(rnn_output,[-1,hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods,output])
loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
epochs = 10000 # 10000 kali iterasi

# Setting proses prediksi dengan acuan perhitungan Mean Standard Error
with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X : x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X : x_batches, y: y_batches})
            print(ep,"\tMSE", mse)
            
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print (y_pred)

#Create graph
plt.title("Forecast vs Actual", fontsize = 14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize = 10, label = 'Actual')
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize = 10, label = 'Forecast')
plt.legend (loc = "upper left")
plt.xlabel ('Time Periods')
plt.show()


