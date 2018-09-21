import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torch.optim as op
import torchvision.transforms as transforms
import torchvision.utils as utils 
from torch.autograd import Variable
import os
import gan
import numpy as np
import random 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, default = 'mnist', help='only mnist supported so far')
parser.add_argument('--dataRoot', default='./mnist', help='where our training and test dataset will be')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='(height, width) of the input image to network')
parser.add_argument('--dimNoise', type=int, default=100, help='size of the latent z vector, or the noise as input to gen')
parser.add_argument('--numFiltersG', type=int, default=128)
parser.add_argument('--numFiltersD', type=int, default=128)
parser.add_argument('--k', type=int, default=1, help='how many static training of {gen,disc} before switching')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0001')
parser.add_argument('--c', type=int, default=1, help='numChannels - unsure')
parser.add_argument('--b1', type=float, default=.5, help='beta1 for Adam')
parser.add_argument('--b2', type=float, default=.999, help='beta2 for Adam')
parser.add_argument('--saveInterval', default=1000, help='interval number we save after')
parser.add_argument('--imageRoot', default= './images', help='images sampled from test data')

hp = parser.parse_args()
print(hp)


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

if hp.dataset == 'mnist':
	if not os.path.exists(hp.dataRoot):
			os.mkdir(hp.dataRoot)

	trans = transforms.Compose([
							   transforms.Resize(hp.imageSize),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_set = dset.MNIST(root=hp.dataRoot, train=True, transform=trans, download=True)
	test_set = dset.MNIST(root=hp.dataRoot, train=False, transform=trans, download=True)

	train_loader = torch.utils.data.DataLoader(
				 dataset=train_set,
				 batch_size=hp.batchSize,
				 shuffle=True)
	test_loader = torch.utils.data.DataLoader(
				dataset=test_set,
				batch_size=hp.batchSize,
				shuffle=False)
print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
print ('==>>> total testing batch number: {}'.format(len(test_loader)))

if not os.path.exists(hp.imageRoot):
	os.mkdir(hp.imageRoot)

gen = gan.Gen(1, hp.numFiltersG, hp.dimNoise)
dis = gan.Dis(1, hp.numFiltersD)


#TO DO: add momen????
gOpt = op.Adam(gen.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))
dOpt = op.Adam(dis.parameters(), lr=hp.lr,betas=(hp.b1, hp.b2))

loss_fn = nn.BCELoss()

for epoch in range(hp.niter):

	for itr, (imgsReal, _) in enumerate(train_loader):
				#options: resize seems to put in Nan Numbers if dim not accounted for?
				#or can grayscale it to increase channel number.
				#or can try to copy elements across all c channels
				z = Variable(torch.randn(hp.batchSize, hp.dimNoise, 1, 1))
				dOpt.zero_grad()
	  		
				actualDReal= dis(imgsReal)
				expectedDReal = Variable(torch.FloatTensor(actualDReal.size()[0],1).fill_(random.uniform(.9, 1.0)), requires_grad=False)
				lossReal = loss_fn(actualDReal, expectedDReal)


				imgGFake = gen(z)
				actualDFake = dis(imgGFake)
				expectedDFake = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.0, 0.1)), requires_grad=False)
				lossFake = loss_fn(actualDFake, expectedDFake)

				dLoss = (lossFake + lossReal) * .5
				dLoss.backward()
				dOpt.step()



				gOpt.zero_grad()
				imgGFake2 = gen(z)

				actualGFake = dis(imgGFake2)
				expectedGFake = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.9, 1.0)), requires_grad=False)
				gLoss = loss_fn(actualGFake, expectedGFake)

				gLoss.backward()
				gOpt.step()

				if (itr % 50) == 0:
					print("Epoch: [%2d] Iter: [%2d] D_loss: %.8f, G_loss: %.8f" % (epoch, itr, dLoss.item(), gLoss.item()))
						


				if (itr % hp.saveInterval) == 0:
					z2 = Variable(torch.randn(hp.batchSize, hp.dimNoise, 1, 1))
					imgFakeTest = gen(z2)
					actualFakeTest = dis(imgFakeTest)
					expectedFakeTest = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.0, 0.1)), requires_grad=False)
					lossFakeTest = loss_fn(actualFakeTest, expectedFakeTest)

					for itr2, (imgsRealTest, _) in enumerate(test_loader):
						if itr2 < 1: 
							actualRealTest= dis(imgsRealTest)
							expectedRealTest = Variable(torch.FloatTensor(actualRealTest.size()[0],1).fill_(random.uniform(.9,1.0)), requires_grad=False)
							lossRealTest = loss_fn(actualRealTest, expectedRealTest)
							print("Testing loss: [%2d]" % (.5 * (lossRealTest + lossFakeTest)))
							utils.save_image(imgsRealTest,'images/testSampleReal_%03d_%03d.png' % (epoch, itr), normalize=True)
							utils.save_image(imgFakeTest.detach(), 'images/testSampleFake_%03d_%03d.png' % (epoch, itr), normalize=True)
