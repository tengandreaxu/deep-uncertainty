import torch
from typing import Tuple


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_test_accuracy(
    model: torch.nn.Module, testloader: torch.utils.data.dataloader.DataLoader
) -> float:
    """returns the test accuracy given a model and testlaoder"""
    test_correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / total
    return test_accuracy


def get_labels_from_dataloader(
    dataloader: torch.utils.data.dataloader.DataLoader,
) -> torch.Tensor:
    """returns all labels from dataloader"""
    labels = []
    for data in dataloader:
        _, label = data
        labels.append(label)
    return torch.cat(labels, 0)


def print_train_step(
    epoch: int,
    batch: int,
    labels: torch.Tensor,
    loss: float,
    outputs: torch.Tensor,
    correct: int,
    total_labels: int,
) -> Tuple[float, float]:
    breakpoint()
    _, train_correct = torch.max(outputs.data, 1)
    correct += (train_correct == labels).sum().item()
    total_labels += len(labels)
    train_acc = correct * 100 / total_labels
    print(
        f"Epoch: {epoch} Batch: {batch} \t Current Loss: {loss:.3f} \t Train Accuracy: {train_acc:.3f}%"
    )
    return correct, total_labels


def print_end_epoch_step(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
):
    print(
        f"Epoch: {epoch} \t Training Loss: {train_loss:.3f} \t Train Accuracy: {train_accuracy:.3f}% \t Val Loss: {val_loss:.3f} Val Accuracy: {val_accuracy:.3f}"
    )
