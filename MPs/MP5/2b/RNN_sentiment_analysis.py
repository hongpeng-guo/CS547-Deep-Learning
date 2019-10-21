import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from RNN_model import RNN_model


def main(input_optimizer, input_batch_size, input_hidden_units, input_epochs):

	glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
	vocab_size = 100000

	x_train = []
	with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		line = line.split(' ')
		line = np.asarray(line,dtype=np.int)

		line[line>vocab_size] = 0

		x_train.append(line)
	x_train = x_train[0:25000]
	y_train = np.zeros((25000,))
	y_train[0:12500] = 1

	x_test = []
	with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		line = line.split(' ')
		line = np.asarray(line,dtype=np.int)

		line[line>vocab_size] = 0

		x_test.append(line)
	y_test = np.zeros((25000,))
	y_test[0:12500] = 1

	vocab_size += 1
	# no_hidden_units = 500
	no_hidden_units = input_hidden_units
	model = RNN_model(no_hidden_units)
	model.cuda()


	# opt = 'sgd'
	# LR = 0.01
	# opt = 'adam'
	opt = input_optimizer
	LR = 0.001
	if(opt=='adam'):
		optimizer = optim.Adam(model.parameters(), lr=LR)
	elif(opt=='sgd'):
		optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

	# batch_size = 200
	batch_size = input_batch_size
	# no_of_epochs = 20
	no_of_epochs = input_epochs
	L_Y_train = len(y_train)
	L_Y_test = len(y_test)

	model.train()

	train_loss = []
	train_accu = []
	test_accu = []

	print ("Optmizer: %s" % opt, "LR: %.6f" % LR, "EpochSize: %d" % no_of_epochs, "BatchSize: %d" % batch_size, "VocalSize: %d" % (vocab_size-1), "HidenSize: %d" % no_hidden_units)


	for epoch in range(no_of_epochs):

		# training
		model.train()

		epoch_acc = 0.0
		epoch_loss = 0.0

		epoch_counter = 0

		time1 = time.time()
		
		I_permutation = np.random.permutation(L_Y_train)

		for i in range(0, L_Y_train, batch_size):

			x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
			sequence_length = 100
			x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
			for j in range(batch_size):
				x = np.asarray(x_input2[j])
				sl = x.shape[0]
				if(sl < sequence_length):
					x_input[j,0:sl] = x
				else:
					start_index = np.random.randint(sl-sequence_length+1)
					x_input[j,:] = x[start_index:(start_index+sequence_length)]
			x_input = glove_embeddings[x_input]
			y_input = y_train[I_permutation[i:i+batch_size]]

			data = Variable(torch.LongTensor(x_input)).cuda()
			target = Variable(torch.FloatTensor(y_input)).cuda()

			optimizer.zero_grad()
			loss, pred = model(data,target,train=True)
			loss.backward()
			
			optimizer.step()   # update weights
			
			prediction = pred >= 0.0
			truth = target >= 0.5
			acc = prediction.eq(truth).sum().cpu().data.numpy()
			epoch_acc += acc
			epoch_loss += loss.data.item()
			epoch_counter += batch_size

		epoch_acc /= epoch_counter
		epoch_loss /= (epoch_counter/batch_size)

		train_loss.append(epoch_loss)
		train_accu.append(epoch_acc)

		print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

		if ((epoch+1)%3) != 0:
			continue

		# ## test
		model.eval()

		epoch_acc = 0.0
		epoch_loss = 0.0

		epoch_counter = 0

		time1 = time.time()
		
		I_permutation = np.random.permutation(L_Y_test)

		for i in range(0, L_Y_test, batch_size):

			x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
			sequence_length = 200
			x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
			for j in range(batch_size):
				x = np.asarray(x_input2[j])
				sl = x.shape[0]
				if(sl < sequence_length):
					x_input[j,0:sl] = x
				else:
					start_index = np.random.randint(sl-sequence_length+1)
					x_input[j,:] = x[start_index:(start_index+sequence_length)]
			y_input = y_train[I_permutation[i:i+batch_size]]

			data = Variable(torch.LongTensor(x_input)).cuda()
			target = Variable(torch.FloatTensor(y_input)).cuda()
			
			with torch.no_grad():
				loss, pred = model(data, target)

			prediction = pred >= 0.0
			truth = target >= 0.5
			acc = prediction.eq(truth).sum().cpu().data.numpy()
			epoch_acc += acc
			epoch_loss += loss.data.item()
			epoch_counter += batch_size

		epoch_acc /= epoch_counter
		epoch_loss /= (epoch_counter/batch_size)

		test_accu.append(epoch_acc)

		time2 = time.time()
		time_elapsed = time2 - time1

		print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

	torch.save(model,'rnn.model')
	data = [train_loss,train_accu,test_accu]
	data = np.asarray(data)
	np.save('data.npy',data)

	os.system('python RNN_test.py')

if __name__ == "__main__":
	main(input_optimizer='adam', input_batch_size=200, input_hidden_units=500, input_epochs=6)
	main(input_optimizer='adam', input_batch_size=200, input_hidden_units=300, input_epochs=6)
	main(input_optimizer='adam', input_batch_size=200, input_hidden_units=800, input_epochs=10)
	main(input_optimizer='adam', input_batch_size=500, input_hidden_units=800, input_epochs=10)
