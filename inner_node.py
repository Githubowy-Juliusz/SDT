import torch
import torch.nn as nn
from leaf import Leaf


class InnerNode(nn.Module):
	def __init__(self, depth: int, max_depth: int,
		input_dim: int, output_dim: int, lmbda: float, device: str):
		super().__init__()

		self.filter = nn.Linear(input_dim, 1).to(device)
		self.beta = nn.Parameter(torch.rand(1)).to(device)

		self.lmbda = lmbda * (2 ** (-depth))
		self.alpha = torch.tensor(0)

		if depth < max_depth:
			self.left_node = InnerNode(depth + 1, max_depth,
				input_dim, output_dim, lmbda, device)
			self.right_node = InnerNode(depth + 1, max_depth,
				input_dim, output_dim, lmbda, device)
		else:
			self.left_node = Leaf(output_dim, device)
			self.right_node = Leaf(output_dim, device)

	def forward(self, x):
		x = self.beta * self.filter(x)
		return torch.sigmoid(x)

	def set_path_probability(self, x, path_probability):
		left_probability = self(x)
		right_probability = 1 - left_probability

		self.alpha = (torch.sum(left_probability * path_probability) / torch.sum(path_probability))

		self.left_node.set_path_probability(x, left_probability * path_probability)
		self.right_node.set_path_probability(x, right_probability * path_probability)

	@property
	def nodes(self) -> list:
		return [self] + self.left_node.nodes + self.right_node.nodes

	@property
	def leaves(self) -> list:
		return self.left_node.leaves + self.right_node.leaves