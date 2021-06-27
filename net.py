import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(144, 512)
		self.fc2 = nn.Linear(512, 10)
	
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.conv3(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 144)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x