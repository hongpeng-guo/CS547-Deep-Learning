import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


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
			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 16, 16)),
			nn.LeakyReLU(),)
		
		self.conv_layer4 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer5 = nn.Sequential(
			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer6 = nn.Sequential(
			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer7 = nn.Sequential(
			nn.Conv2d(3, 196, kernel_size=3, padding=1, stride=1),
			nn.LayerNorm((196, 8, 8)),
			nn.LeakyReLU(),)
		
		self.conv_layer8 = nn.Sequential(
			nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=2),
			nn.LayerNorm((196, 4, 4)),
			nn.LeakyReLU(),)

		self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4),

		self.fc1 = nn.Linear(196, 1)

		self.fc10 = nn.Linear(196, 10)
		

	def forward(self, x):
		conv_out_1 = self.conv_layer1(x)
		conv_out_2 = self.conv_layer2(conv_out_1)
		conv_out_3 = self.conv_layer3(conv_out_2)
		conv_out_4 = self.conv_layer4(conv_out_3)
		conv_out_5 = self.conv_layer5(conv_out_4)
		conv_out_6 = self.conv_layer6(conv_out_5)
		conv_out_7 = self.conv_layer7(conv_out_6)
		conv_out_8 = self.conv_layer8(conv_out_7)
		pool_out = self.max_pool(conv_out_8)
		pool_out = pool_out.reshape(pool_out.size(0), -1)
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