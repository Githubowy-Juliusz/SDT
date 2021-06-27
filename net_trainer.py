import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTrainer:
	def __init__(self, model, device, learning_rate=0.001, weight_decay=0.05):
		self.model = model
		self.device = device
		self.loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.AdamW(self.model.parameters(), 
			lr=learning_rate, weight_decay=weight_decay)

	def train(self, data_loader):
		self.model.train()
		accuracies = []
		losses = []
		for data, labels in data_loader:
			data = data.to(self.device)
			labels = labels.to(self.device)

			self.optimizer.zero_grad()

			predictions = self.model(data)
			loss = self.loss_fn(predictions, labels)
			losses.append(loss.detach().cpu().numpy())

			loss.backward()
			self.optimizer.step()

			predictions = F.softmax(predictions, dim=1)
			accuracy = (torch.argmax(predictions, dim=1) == labels).type(torch.FloatTensor).mean().item()
			accuracies.append(accuracy)

		return np.mean(accuracies), np.mean(losses)

	def validate(self, data_loader):
		self.model.eval()
		losses = []
		accuracies = []
		for data, labels in data_loader:
			data = data.to(self.device)
			labels = labels.to(self.device)

			predictions = self.model(data)
			loss = self.loss_fn(predictions, labels)
			losses.append(loss.detach().cpu().numpy())

			predictions = F.softmax(predictions, dim=1)
			accuracy = (torch.argmax(predictions, dim=1) == labels).type(torch.FloatTensor).mean().item()
			accuracies.append(accuracy)

		return np.mean(accuracies), np.mean(losses)