import torch
import torch.nn as nn
import torchvision.datasets as dset
import utils 

class Gen(nn.Module):
	def __init__(self, c, numFilters, dimNoise): 
		super(Gen, self).__init__()

		self.dimNoise = dimNoise
		self.numFiltersG = numFilters
		self.model = nn.Sequential(
			nn.Linear(1,1, bias = False),
			nn.ConvTranspose2d(dimNoise, numFilters * 4, 4, 1, 0, bias=False),
			nn.BatchNorm2d(numFilters * 4, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),
			
			nn.ConvTranspose2d(numFilters * 4, numFilters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 2, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),

			nn.ConvTranspose2d(numFilters * 2, numFilters, 2, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),

			nn.ConvTranspose2d(    numFilters,      c, 4, 2, 1, bias=False),
			nn.Linear(28,28,bias=False),
			nn.Tanh()
			)			
		utils.initialize_weights(self)

	def forward(self, z):
		return self.model(z)


class Dis(nn.Module):
	def __init__(self, c, numFilters): 
		super(Dis, self).__init__()
		self.numFiltersD = numFilters

		self.model = nn.Sequential(
			nn.Linear(28,28, bias=False),
			nn.Conv2d(c, numFilters, 4, 2, 1, bias=False),
			nn.LeakyReLU(.2, inplace=True),

			nn.Conv2d(numFilters, numFilters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 2, momentum = .2),
			nn.LeakyReLU(.2, inplace=True),

			nn.Conv2d(numFilters * 2, numFilters * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 4, momentum = .2),
			nn.LeakyReLU(.2, inplace=True),

			nn.Conv2d(numFilters * 4,     1, 4, 2, 1, bias=False),
			
			nn.Linear(1,1, bias=False),
			nn.Sigmoid()
			)			
		utils.initialize_weights(self)

	def forward(self, img):
		x = self.model(img)
		return x.view(-1,1)





