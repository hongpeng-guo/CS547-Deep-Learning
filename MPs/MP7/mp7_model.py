import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class discriminator(nn.Module):
	def __init__(self, num_classes=10):
		super(discriminator, self).__init__()

		self.conv_layer = nn.Sequential(

			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm(196),
			nn.LeakyReLU(),

			nn.MaxPool2d(kernel_size=4, stride=4),
		)
		
		self.fc1 = nn.Linear(196, 1)
		self.fc10 = nn.Linear(196, 10)
		

	def forward(self, x):
		conv_out = self.conv_layer(x)
		conv_out = conv_out.reshape(conv_out.size(0), -1)
		fc1_out = self.fc1(conv_out)
		fc10_out = self.fc10(conv_out)
		return fc1_out, fc10_out



class generator(nn.Module):
	def __init__(self, num_classes=10):
		super(ConvNet, self).__init__()

		self.fc1 = nn.Linear(100, 196*4*4)

		self.conv_layer = nn.Sequential(

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm(196),
			nn.ReLU(),

			nn.ConvTranspose2d(196, 3, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm(3),
			nn.ReLU(),
		)

		
	def forward(self, x):
		fc1_out = self.fc1(x)
		fc1_out = fc1_out.reshape(-1, 196, 3, 3)
		conv_out = self.conv_layer(fc1_out)
		return conv_out