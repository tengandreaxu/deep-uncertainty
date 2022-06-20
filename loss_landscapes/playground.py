"""
This script is used to test models accuracy, correctness, etc.
"""
import torch
import torchvision
from loss_landscapes.models.MediumCNN import MediumCNN
from plotting.Plotter import Plotter
from helpers.DatasetsManager import DatasetsManager
from utils.pytorch_custom import get_test_accuracy

if __name__ == "__main__":

    cnn = MediumCNN()
    cnn.load_state_dict(torch.load("saved_models/mediumCNN"))

    plotter = Plotter()
    dm = DatasetsManager()

    _, testloader = dm.torch_load_cifar_10()

    test_accuracy = get_test_accuracy(cnn, testloader=testloader)

    print(f"Test Accuracy: {test_accuracy:.3f}%")

    # check images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    to_plot = torchvision.utils.make_grid(images)
    breakpoint()
    classes = dm.get_cifar_10_label_names()
    plotter.pytorch_imshow(to_plot)
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]}" for j in range(4)))
    breakpoint()
