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
H, W = 28, 28
H_k, W_k, C = 4, 4, 8
num_outputs = 10
batch_size = 1

model = {}
model['K'] = np.random.randn(H_k, W_k, C)/np.sqrt(H_k * W_k * C)
model['b'] = np.random.randn(num_outputs, 1)
model['W'] = np.random.randn(num_outputs, H-H_k+1, W-W_k+1, C)/np.sqrt(num_outputs * (H-H_k+1) * (W-W_k+1) * C)
grads = copy.deepcopy(model)

print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Traing Model
import time
time1 = time.time()
LR = .1
num_epochs = 6
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
	e_time1 = time.time()
	for n in range(x_train.shape[1] // batch_size):
		'''
		Make Input Y shape to be (num_outputs, batch_size)
		Make Input X shape to be (batch_size, H, W)
		'''
		y = y_train[:, batch_size*n: batch_size*n+batch_size]
		x = x_train[:, batch_size*n: batch_size*n+batch_size]
		x = x.T.reshape(batch_size, H, W)
		cache = feed_forward(x, model)
		prediction = np.argmax(cache['R'], axis=0)
		total_correct += np.sum(prediction == np.argmax(y, axis=0))
		grads = back_propagate(x, y, model, cache)
		model['W'] = model['W'] - LR * grads['W']
		model['K'] = model['K'] - LR * grads['K']
		model['b'] = model['b'] - LR * grads['b']
	e_time2 = time.time()
	print(epochs, e_time2 - e_time1, total_correct/np.float(x_train.shape[1]))
time2 = time.time()
print(time2-time1)

#Testing Data
total_correct = 0
for n in range( x_test.shape[1] ):
	y = np.expand_dims(y_test[:, n], axis=1)
	x = np.expand_dims(x_test[:, n], axis=1)
	x = x.T.reshape(1, H, W)
	cache = feed_forward(x, model)
	prediction = np.argmax(cache['R'])
	if (prediction == np.argmax(y)):
		total_correct += 1
print(total_correct/np.float(x_test.shape[1]) )
