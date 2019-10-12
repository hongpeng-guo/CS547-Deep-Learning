import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


num_epochs = 40
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

# This is the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
# Your own directory to the train folder of tiyimagenet
train_dir = dir_path + '/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dir = dir_path + '/tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


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
		self.curt_in_size = 64

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
		self.maxpool = nn.MaxPool2d(4, stride=4)
		self.curt_in_size = self.curt_in_size // 4

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
model = ResNet(BasicBlock, [2, 4, 4, 2], 200).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(num_epochs):
	# Count scheduler step
	scheduler.step()
	# Train the model
	model.train()
	total_step = len(train_loader)
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
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
		
	print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
	   

	# Test the model
	with torch.no_grad():
		model.eval()
		total, correct = 0, 0
		for batch_idx, (X_test_batch, Y_test_batch) in enumerate(val_loader):
			X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
			outputs = model(X_test_batch)
			_, predicted = torch.max(outputs.data, 1)
			total += Y_test_batch.size(0)
			correct += (predicted == Y_test_batch).sum().item()

	print('Test Accuracy of the model: {} %'.format(100 * correct / total))


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

