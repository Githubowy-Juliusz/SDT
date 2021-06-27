import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from leaf import Leaf
from inner_node import InnerNode


class DecisionTree(nn.Module):
	def __init__(self, max_depth: int, input_dim: int,
		output_dim: int, device: str, lmbda=0.1):
		super().__init__()

		self.output_dim = output_dim
		self.input_dim = input_dim
		self.device = device

		self.root = InnerNode(1, max_depth, input_dim, output_dim, lmbda, device)
		self.nodes = self.root.nodes
		self.leaves = self.root.leaves

	def forward(self, x):
		empty_path_probability = torch.ones((len(x), 1)).to(self.device)
		self.root.set_path_probability(x, empty_path_probability)
		return self._predict_classes(len(x))

	def _predict_classes(self, batch_size):
		#pick leaf with highest path probability for given sample
		leaves = [max(self.leaves, key=lambda leaf: leaf.path_probability[i])
			for i in range(batch_size)]
		#calculate probabilities of a given class
		predictions = [leaf.forward() for leaf in leaves]
		#pick class with highest probability
		return [np.argmax(pred.detach().cpu().numpy()) for pred in predictions]