import numpy as np
import h5py
import time
import copy
from random import randint
from utils import *

# Load MNIST Data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

x_train, x_test = x_train.T, x_test.T
y_train, y_test = np.eye(10)[y_train].T, np.eye(10)[y_test].T


# Implementation of SGD Algorithm
num_inputs = 28*28
num_hiddens = 128
num_outputs = 10
batch_size = 5

model = {}
model['W'] = np.random.randn(num_hiddens,num_inputs)/np.sqrt(num_inputs)
model['b1'] = np.random.randn(num_hiddens)
model['C'] = np.random.randn(num_outputs, num_hiddens)/np.sqrt(num_hiddens)
model['b2'] = np.random.randn(num_outputs)
grads = copy.deepcopy(model)

print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Traing Model
import time
time1 = time.time()
LR = .1
num_epochs = 40
for epochs in range(num_epochs):
	#Learning rate schedule
	if (epochs > 10):
		LR = 0.01
	if (epochs > 20):
		LR = 0.001
	if (epochs > 30):
		LR = 0.0001
	total_correct = 0
	random_order = np.random.permutation(x_train.shape[1])
	x_train, y_train = x_train[:, random_order], y_train[:, random_order]
	for n in range(x_train.shape[1] // batch_size):
		y = y_train[:, batch_size*n: batch_size*n+batch_size]
		x = x_train[:, batch_size*n: batch_size*n+batch_size]
		cache = feed_forward(x, model)
		prediction = np.argmax(cache['R'], axis=0)
		total_correct += np.sum(prediction == np.argmax(y, axis=0))
		grads = back_propagate(x, y, model, cache)
		model['W'] = model['W'] - LR * grads['W']
		model['C'] = model['C'] - LR * grads['C']
		model['b1'] = model['b1'] - LR * grads['b1']
		model['b2'] = model['b2'] - LR * grads['b2']
	print(total_correct/np.float(x_train.shape[1]))
time2 = time.time()
print(time2-time1)

#Testing Data
total_correct = 0
for n in range( x_test.shape[1] ):
	y = np.expand_dims(y_test[:, n], axis=1)
	x = np.expand_dims(x_test[:, n], axis=1)
	cache = feed_forward(x, model)
	prediction = np.argmax(cache['R'])
	if (prediction == np.argmax(y)):
		total_correct += 1
print(total_correct/np.float(x_test.shape[1]) )
