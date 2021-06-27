import numpy as np
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_custom_digits(file_path: str, batch_size: int):
	img = cv.imread(file_path)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	cell_width = 101
	cell_height = 100.285714286
	padding_width = 6
	padding_height = 7

	images = []
	labels = []
	one_hot_encoded_labels = []
	label_counter = 0

	for y in np.arange(0, img.shape[0], cell_height + padding_height):
		for x in range(0, img.shape[1], cell_width + padding_width):
			images.append(img[int(y):int(y + cell_height), int(x):int(x + cell_width)])
			labels.append(label_counter % 10)
			label = np.zeros(10)
			label[label_counter % 10] = 1
			one_hot_encoded_labels.append(label)
		label_counter += 1

	images = [cv.resize(img, (28, 28), interpolation=cv.INTER_NEAREST) for img in images]
	images = [cv.bitwise_not(img) for img in images]
	images = np.array(images)
	labels = np.array(labels)
	one_hot_encoded_labels = np.array(one_hot_encoded_labels)

	dataset = CustomDigitsDataset(images, labels)
	dataset_one_hot = CustomDigitsDataset(images, one_hot_encoded_labels)
	return (DataLoader(dataset, batch_size=batch_size, shuffle=True), 
		DataLoader(dataset_one_hot, batch_size=batch_size, shuffle=True))

class CustomDigitsDataset(Dataset):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])

	def __len__(self):
		return (len(self.images))

	def __getitem__(self, index):
		image = self.images[index]
		label = self.labels[index]
		return self.transforms(image), label