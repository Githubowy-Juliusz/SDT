import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_onehot_encoded_MNIST_dataloaders(batch_size: int,
	val_percentage=0.2, num_of_classes=10) -> tuple:
	train, val, test = _get_MNIST(val_percentage)

	train = _encode_labels(train, num_of_classes)
	val = _encode_labels(val, num_of_classes)
	test = _encode_labels(test, num_of_classes)

	return _create_dataloaders(train, val, test, batch_size)

def get_MNIST_dataloaders(batch_size: int, val_percentage=0.2) -> tuple:
	return _create_dataloaders(*_get_MNIST(val_percentage), batch_size)

def create_soft_label_MNIST_dataloader(model: nn.Module,
	dataloader: DataLoader, device: str) -> DataLoader:
	new_data = []
	new_labels = []
	for data, label in dataloader:
		new_data.append(data)

		data = data.to(device)
		predictions = F.softmax(model(data), dim=1).detach().cpu()
		new_labels.append(predictions)

	new_data = torch.stack(new_data).view(-1, 1, 28, 28)
	new_labels = torch.stack(new_labels).squeeze().view(-1, 10)

	return DataLoader(TensorDataset(new_data, new_labels), dataloader.batch_size)

def _get_MNIST(val_percentage) -> tuple:
	directory, _ = os.path.split(os.path.realpath(__file__))
	mnist_path = f"{directory}/MNIST/"

	mnist_train = MNIST(root=mnist_path, train=True, download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		]))

	mnist_test = MNIST(root=mnist_path, train=False, download=True, 
		transform=transforms.Compose([
			transforms.ToTensor(), 
			transforms.Normalize((0.1307,), (0.3081,))
		]))

	val_size = int(len(mnist_train) * val_percentage)
	train_size = len(mnist_train) - val_size

	mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

	return mnist_train, mnist_val, mnist_test

def _create_dataloaders(mnist_train, mnist_val, mnist_test, batch_size) -> tuple:
	data_loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
	data_loader_val = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
	data_loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

	return data_loader_train, data_loader_val, data_loader_test

def _onehot_encoding(data, num_of_classes) -> torch.Tensor:
	zeros = torch.zeros(data.size()[0], num_of_classes)
	return zeros.scatter(dim=1, index=data.view(-1, 1), value=1)

def _encode_labels(dataset, num_of_classes) -> TensorDataset:
	new_data = []
	new_labels = []
	for data, label in dataset:
		new_data.append(data)
		new_labels.append(_onehot_encoding(torch.tensor([label]), num_of_classes))

	new_data = torch.stack(new_data)
	new_labels = torch.stack(new_labels).squeeze()
	return TensorDataset(new_data, new_labels)