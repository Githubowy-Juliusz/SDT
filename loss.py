import torch


class SoftDecisionTreeLoss:
	def __init__(self, leaves, nodes):
		self.leaves = leaves
		self.nodes = nodes

	def __call__(self, labels):
		loss = 0
		for leaf in self.leaves:
			Q = torch.transpose(leaf(), 0, 1).double()
			loss += leaf.path_probability * torch.matmul(labels.double(), torch.log(Q))
		C = self._calculate_penalty()
		return -loss.mean() + C

	def _calculate_penalty(self):
		C = 0
		for node in self.nodes:
			C += -node.lmbda * 0.5 *(torch.log(node.alpha) + torch.log(1 - node.alpha))
		return C