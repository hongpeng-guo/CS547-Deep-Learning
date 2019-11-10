import torch 
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time


class discriminator(nn.Module):
	def __init__(self, num_classes=10):
		super(discriminator, self).__init__()

		self.conv_layer1 = nn.Sequential(
			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 32, 32)),
			nn.LeakyReLU(),)

		self.conv_layer2 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm((196, 16, 16)),
			nn.LeakyReLU(),)
		
		self.conv_layer3 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 16, 16)),
			nn.LeakyReLU(),)
		
		self.conv_layer4 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer5 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer6 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer7 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer8 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm((196, 4, 4)),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=4, stride=4),)

		self.fc1 = nn.Linear(196, 1)

		self.fc10 = nn.Linear(196, 10)
		

	def forward(self, x, extract_features=0):
		conv_out_1 = self.conv_layer1(x)
		conv_out_2 = self.conv_layer2(conv_out_1)
		conv_out_3 = self.conv_layer3(conv_out_2)
		conv_out_4 = self.conv_layer4(conv_out_3)
		if(extract_features==4):
			x = conv_out_4
			x = F.max_pool2d(x,8,8)
			x = x.view(-1, 196)
			return x
		conv_out_5 = self.conv_layer5(conv_out_4)
		conv_out_6 = self.conv_layer6(conv_out_5)
		conv_out_7 = self.conv_layer7(conv_out_6)
		conv_out_8 = self.conv_layer8(conv_out_7)
		if(extract_features==8):
			x = conv_out_8
			return x		
		conv_out_8 = conv_out_8.reshape(conv_out_8.size(0), -1)
		fc1_out = self.fc1(conv_out_8)
		fc10_out = self.fc10(conv_out_8)
		return fc1_out, fc10_out



class generator(nn.Module):
	def __init__(self, num_classes=10):
		super(generator, self).__init__()

		self.fc1 = nn.Sequential(
			nn.Linear(100, 196*4*4),
			nn.BatchNorm1d(196*4*4),
		)

		self.conv_layer = nn.Sequential(

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm2d(196),
			nn.ReLU(),

			nn.Conv2d(196, 3, kernel_size=3, padding=1, stride=1),
		)

		
	def forward(self, x):
		fc1_out = self.fc1(x)
		fc1_out = fc1_out.reshape(-1, 196, 4, 4)
		conv_out = self.conv_layer(fc1_out)
		conv_out = torch.tanh(conv_out)
		return conv_out


n_z = 100
n_classes = 10
gen_train = 1
num_epochs = 200
batch_size = 128


def plot(samples):
	fig = plt.figure(figsize=(10, 10))
	gs = gridspec.GridSpec(10, 10)
	gs.update(wspace=0.02, hspace=0.02)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample)
	return fig


transform_test = transforms.Compose([
	transforms.CenterCrop(32),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('discriminator.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
	output = model(X, extract_features=4)

	loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
	gradients = torch.autograd.grad(outputs=loss, inputs=X,
							  grad_outputs=torch.ones(loss.size()).cuda(),
							  create_graph=True, retain_graph=False, only_inputs=True)[0]

	prediction = output.data.max(1)[1] # first column has actual prob.
	accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
	print(i,accuracy,-loss)

	X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
	X[X>1.0] = 1.0
	X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/w_max_features_L4.png', bbox_inches='tight')
plt.close(fig)