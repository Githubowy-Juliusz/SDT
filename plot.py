import matplotlib.pyplot as plt

_figsize = (30, 15)

def plot_accuracies(accuracies_train, accuracies_val):
	plt.figure(figsize=_figsize)
	plt.rcParams.update({"font.size": 20})

	plt.plot(accuracies_train, color="green", label="Train accuracy")
	plt.plot(accuracies_val, color="blue", label="Val accuracy")
	plt.axhline(1, color="black", label="1", linestyle="dashed")

	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("Accuracies")
	plt.show()

def plot_losses(losses_train, losses_val):
	plt.figure(figsize=_figsize)
	plt.rcParams.update({"font.size": 20})

	plt.plot(losses_train, color="green", label="Train loss")
	plt.plot(losses_val, color="blue", label="Val loss")

	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Losses")
	plt.show()

def plot_mnist(loader):
	fig = plt.figure(figsize=_figsize)
	rows = 2
	columns = 5
	position = 1

	for images, labels in loader:
		images = images[:10]
		for image in images:
			fig.add_subplot(rows, columns, position)
			plt.imshow(image.permute(1, 2, 0), cmap="gray")
			position += 1
		break
	plt.plot()