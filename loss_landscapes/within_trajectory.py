import torch
import torch.nn as nn
from loss_landscapes.models.MediumCNN import MediumCNN
from helpers.DatasetsManager import DatasetsManager

torch.set_num_threads(10)
if __name__ == "__main__":
    dm = DatasetsManager()
    cnn = MediumCNN()

    trainloader, testloader = dm.torch_load_cifar_10(batch_size=cnn.batch_size)
    classes = dm.get_cifar_10_label_names()
    lr = cnn.learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=lr)

    for epoch in range(cnn.epochs):

        epoch_number = epoch + 1
        current_loss = 0.0
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

            loss_print = loss.item()

            _, train_correct = torch.max(outputs.data, 1)
            correct += (train_correct == labels).sum().item()
            total_labels += len(labels)
            train_acc = correct * 100 / total_labels

            print(
                f"Epoch: {epoch} Batch: {i} \t Current Loss: {loss_print:.3f} \t Train Accuracy: {train_acc:.3f}%"
            )
            break
        train_accuracy = 100 * correct / len(trainloader)

        test_correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = cnn(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        breakpoint()
        print(
            f"Epoch: \t {epoch} \t Train Accuracy: \t {train_accuracy:.3f} \t Test Accuracy: \t {test_accuracy:.3f}"
        )
