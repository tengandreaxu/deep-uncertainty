import argparse
import torch
import torch.nn as nn

from loss_landscapes.models.MediumCNN import MediumCNN
from helpers.DatasetsManager import DatasetsManager
from loss_landscapes.models.SmallCNN import SmallCNN
from utils.pytorch_custom import get_test_accuracy, print_train_step

torch.set_num_threads(25)


def train_cifar_10_model(cnn: torch.nn.Module, model_type: str):
    dm = DatasetsManager()

    (
        trainloader,
        valloader,
        testloader,
        train_dss,
        val_ds,
        testset,
    ) = dm.torch_load_cifar_10(batch_size=cnn.batch_size, validation_set=500)
    lr = cnn.learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=lr)

    for epoch in range(cnn.epochs):

        epoch_number = epoch + 1
        correct = 0
        total_labels = 0
        if epoch_number % 10 == 0:
            lr = lr / 2
            optimizer.param_groups[0]["lr"] = lr

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            optimizer.zero_grad()
            outputs = cnn(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct, total_labels = print_train_step(
                epoch, i, labels, loss.item(), outputs, correct, total_labels
            )

        train_accuracy = 100 * correct / total_labels
        test_accuracy = get_test_accuracy(cnn, testloader)
        print(
            f"Epoch: \t {epoch} \t Train Accuracy: \t {train_accuracy:.3f} \t Test Accuracy: \t {test_accuracy:.3f}"
        )

    torch.save(cnn.state_dict(), f"saved_models/{model_type}CNN")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        help="Choose which model to train if 'small' for SmallCNN or 'medium' for MediumCNN",
    )
    parser.set_defaults(model="small")

    args = parser.parse_args()

    if args.model == "small":
        cnn = SmallCNN()
    else:
        cnn = MediumCNN()

    train_cifar_10_model(cnn, model_type=args.model)
