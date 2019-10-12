import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time




# We provide the code for loading CIFAR100 data
num_epochs = 40
batch_size = 128
learning_rate = 0.01

# torch.manual_seed(0)
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# YOUR CODE GOES HERE
class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=False):
		super(BasicBlock, self).__init__()

		self.conv_layer = nn.Sequential(

			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace = True),

			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
		)

		self.relu = nn.ReLU(inplace = True)

		self.downsample = downsample
		if self.downsample:
			self.projection = nn.Sequential(
				nn.Conv2d(in_channels,
						  out_channels,
						  kernel_size=1,
						  stride=stride,
						  padding=0,
						  bias=False),
				nn.BatchNorm2d(out_channels)
		)

	def forward(self, x):
		conv_out = self.conv_layer(x)
		resi_out = x
		if self.downsample:
			resi_out = self.projection(resi_out)
		out = self.relu(conv_out + resi_out)
		return out


class ResNet(nn.Module):
	def __init__(self, basic_block, num_blocks, num_classes):
		super(ResNet, self).__init__()
		
		self.curt_in_channels = 32
		self.curt_in_size = 32

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5)
		)

		self.conv2_x = self._add_layers(32, num_blocks[0])
		self.conv3_x = self._add_layers(64, num_blocks[1], start_stride=2)
		self.conv4_x = self._add_layers(128, num_blocks[2], start_stride=2)
		self.conv5_x = self._add_layers(256, num_blocks[3], start_stride=2)

		self.dropout = nn.Dropout(p=0.5)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(4, stride=1)
		self.curt_in_size = self.curt_in_size - 3

		self.fc = nn.Linear(256 * (self.curt_in_size**2), num_classes)
	
	def forward(self, x):
		x = self.dropout(self.relu(self.conv1(x)))

		x = self.conv2_x(x)
		x = self.conv3_x(x)
		x = self.conv4_x(x)
		x = self.conv5_x(x)

		x = self.maxpool(x)

		x = x.view(x.shape[0], -1)
		x = self.fc(x)

		return x

	def _add_layers(self, out_channels, num_blocks, start_stride=1):
		downsample = False
		if start_stride != 1 or self.curt_in_channels != out_channels:
			downsample = True
			self.curt_in_size = self.curt_in_size // start_stride
		
		layers = []
		layers.append(BasicBlock(self.curt_in_channels, out_channels,
								 stride=start_stride, downsample=downsample))
		self.curt_in_channels = out_channels

		for _ in range(1, num_blocks):
			layers.append(BasicBlock(self.curt_in_channels, out_channels))
			layers.append(nn.Dropout(p=0.2))
		
		return nn.Sequential(*layers)


# Change to your ResNet
model = ResNet(BasicBlock, [2, 4, 4, 2], 100).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 100], gamma=0.1)

for epoch in range(num_epochs):
	# Count scheduler step
	scheduler.step()
	# Train the model
	model.train()
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
		X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
		
		# Forward pass
		outputs = model(X_train_batch)
		loss = criterion(outputs, Y_train_batch)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		# for group in optimizer.param_groups:
		# 	for p in group['params']:
		# 		state = optimizer.state[p]
		# 		if 'step' in state.keys():
		# 			if(state['step']>=1024):
		# 				state['step'] = 1000
		optimizer.step()

	print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

	   
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