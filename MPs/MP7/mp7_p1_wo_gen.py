import torch 
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from mp7_model import discriminator, generator

# Hyper parameters
num_epochs = 100
batch_size = 128


transform_train = transforms.Compose([
	transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
	transforms.ColorJitter(
			brightness=0.1*torch.randn(1),
			contrast=0.1*torch.randn(1),
			saturation=0.1*torch.randn(1),
			hue=0.1*torch.randn(1)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
	transforms.CenterCrop(32),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


model =  discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_accuracy_final = []
test_accuracy_final = []

for epoch in range(num_epochs):
	
	if(epoch==50):
		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate/10.0
	if(epoch==75):
		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate/100.0

	# Train the model
	model.train()
	train_accuracy = []
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
		if(Y_train_batch.shape[0] < batch_size):
			continue
		X_train_batch = Variable(X_train_batch).cuda()
		Y_train_batch = Variable(Y_train_batch).cuda()
		_, output = model(X_train_batch)

		loss = criterion(output, Y_train_batch)
		optimizer.zero_grad()

		loss.backward()
		optimizer.step()

		prediction = output.data.max(1)[1] #Label Prediction 
		accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
		train_accuracy.append(accuracy)   
	accuracy_epoch = np.mean(train_accuracy)
	print('\nIn epoch ', epoch,' the accuracy of the training set =', accuracy_epoch)
	train_accuracy_final.append(accuracy_epoch)

	# Test the model
	model.eval()
	with torch.no_grad():
		test_accu = []
		for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
			X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

			with torch.no_grad():
				_, output = model(X_test_batch)

			prediction = output.data.max(1)[1] # first column has actual prob.
			accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
			test_accu.append(accuracy)
			accuracy_test = np.mean(test_accu)
	print('\nIn epoch ', epoch,' the accuracy of the training set =', accuracy_test)
	train_accuracy_final.append(accuracy_epoch)


print (train_accuracy_final)
print (test_accuracy_final)


torch.save(model, 'cifar10.model')