import numpy as np
import torch
from sklearn.metrics import accuracy_score
from loss import SoftDecisionTreeLoss


class TreeTrainer:
	def __init__(self, model, device, learning_rate=0.001):
		self.model = model
		self.device = device
		self.loss_fn = SoftDecisionTreeLoss(model.leaves, model.nodes)
		self.optimizer = torch.optim.AdamW(self.model.parameters(), 
			lr=learning_rate, weight_decay=0.01)

	def train(self, data_loader):
		self.model.train()
		accuracies = []
		losses = []
		for data, labels in data_loader:
			data = data.view(len(data), -1).to(self.device)
			labels = labels.to(self.device)

			self.optimizer.zero_grad()

			predictions = self.model(data)
			loss = self.loss_fn(labels)
			losses.append(loss.detach().cpu().numpy())

			loss.backward()
			self.optimizer.step()

			predictions = np.array(predictions)
			labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
			accuracies.append(accuracy_score(labels, predictions))

		return np.mean(accuracies), np.mean(losses)

	def validate(self, data_loader):
		self.model.eval()
		losses = []
		accuracies = []
		for data, labels in data_loader:
			data = data.view(len(data), -1).to(self.device)
			labels = labels.to(self.device)

			predictions = self.model(data)
			loss = self.loss_fn(labels)
			losses.append(loss.detach().cpu().numpy())

			predictions = np.array(predictions)
			labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
			accuracies.append(accuracy_score(labels, predictions))

		return np.mean(accuracies), np.mean(losses)