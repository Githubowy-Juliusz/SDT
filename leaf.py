import torch
import torch.nn as nn


class Leaf(nn.Module):
	def __init__(self, output_dim: int, device: str):
		super().__init__()

		self.distribution = nn.Parameter(torch.rand(output_dim)).to(device)
		self.path_probability = torch.tensor(0)

	def forward(self, *args, **kwargs):
		return torch.softmax(self.distribution.view(1, -1), dim=1)

	def set_path_probability(self, x, path_probability):
		self.path_probability = path_probability

	@property
	def nodes(self) -> list:
		return []

	@property
	def leaves(self) -> list:
		return [self]