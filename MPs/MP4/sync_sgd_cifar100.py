import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os, time
import subprocess
from mpi4py import MPI


# Code for iniitialization pytorch distributed 

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
	stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor


# Your code start here 

# We provide the code for loading CIFAR100 data
num_epochs = 20
batch_size = 128
learning_rate = 0.001

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)



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
			layers.append(nn.Dropout(p=0.1))
		
		return nn.Sequential(*layers)

	

model = ResNet(BasicBlock, [2, 4, 4, 2], 100)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Make sure that all nodes have the same model
for param in model.parameters():
	tensor0 = param.data
	dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
	param.data = tensor0/np.sqrt(np.float(num_nodes))
model.cuda()

Path_Save = os.path.dirname(os.path.realpath(__file__))
#torch.save(model.state_dict(), Path_Save)
#model.load_state_dict(torch.load(Path_Save))

def train():
	model.train()
	for batch_idx, (images, labels) in enumerate(trainloader):
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()

		optimizer.zero_grad()
		outputs = net(images)
		loss = criterion(outputs, labels)
		loss.backward()
		for param in net.parameters():
			tensor0 = param.grad.data.cpu()
			dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
			tensor0 /= float(num_nodes)
			param.grad.data = tensor0.cuda()
		optimizer.step()

def eval(dataloader):
	model.eval()
	test_loss = 0.0
	correct = 0.0
	for batch_idx, (images, labels) in enumerate(dataloader):
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()

		outputs = net(images) # 100x100
		loss = criterion(outputs, labels)
		test_loss += loss.data[0]
		_, preds = outputs.max(1)
		cor = preds.eq(labels).sum()
		correct += cor.data[0]
	return test_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

if __name__=='__main__':
	num_epochs = 500
	for epoch in range(num_epochs):
		train()
		test_loss,test_acc = eval(testloader)
		train_loss,train_acc  =eval(trainloader)
		scheduler.step(epoch)
		print('%d\t%d\t%f\t%f\t%f\t%f' % (rank,epoch,test_loss,test_acc,train_loss,train_acc))
		if test_acc > 0.65:
			break

# Save the model checkpoint
torch.save(model.state_dict(), Path_Save)