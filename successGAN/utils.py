import torch.nn as nn

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)