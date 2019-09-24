import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001


# Training data augmentation
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./',
										   train=True, 
										   transform=transform_train,
										   download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./',
										  train=False, 
										  transform=transform_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size, 
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size, 
										  shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
	def __init__(self, num_classes=10):
		super(ConvNet, self).__init__()

		self.conv_layer = nn.Sequential(

			# Conv Layer block 1
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Conv Layer block 2
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(p=0.05),

			# Conv Layer block 3
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.fc_layer = nn.Sequential(
			nn.Dropout(p=0.1),
			nn.Linear(8*8*64 , 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(p=0.1),
			nn.Linear(512, 10)
		)
		
	def forward(self, x):
		conv_out = self.conv_layer(x)
		conv_out = conv_out.reshape(conv_out.size(0), -1)
		fc_out = self.fc_layer(conv_out)
		return fc_out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
model.eval()  
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
