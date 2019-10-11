import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
import time
import os

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


num_epochs = 20
batch_size = 128
learning_rate = 0.01

transform_train = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomCrop(224, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Loading the data
trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dir_path = os.path.dirname(os.path.realpath(__file__))
def resnet18(pretrained=True):
	model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dir_path))
	return model

model = resnet18(pretrained=True)
# for param in model.parameters():
# 	param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(num_epochs):
	# Count scheduler step
	scheduler.step()
	# Train the model
	model.train()
	total_step = len(trainloader)
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
		X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
		
		# Forward pass
		outputs = model(X_train_batch)
		loss = criterion(outputs, Y_train_batch)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		for group in optimizer.param_groups:
			for p in group['params']:
				state = optimizer.state[p]
				if 'step' in state.keys():
					if(state['step']>=1024):
						state['step'] = 1000
		optimizer.step()
		
	print ('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
	   

	# Test the model
	with torch.no_grad():
		model.eval()
		total, correct = 0, 0
		for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
			X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
			outputs = model(X_test_batch)
			_, predicted = torch.max(outputs.data, 1)
			total += Y_test_batch.size(0)
			correct += (predicted == Y_test_batch).sum().item()

	print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
