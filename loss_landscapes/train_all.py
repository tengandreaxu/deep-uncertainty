import os
import torch
import time
import torch.nn as nn
from copy import deepcopy
from helpers.DatasetsManager import DatasetsManager
from loss_landscapes.models.MediumCNN import MediumCNN
from utils.pytorch_custom import print_end_epoch_step, print_train_step
from loss_landscapes.func_utils import save_data
from loss_landscapes.paths import (
    BS_BY_EPOCHS,
    BS_MANY,
    VALIDATION_SET,
    WS_BY_EPOCHS,
    WS_MANY,
    WS_TRAJECTORY,
    BS_TRAJECTORY,
    OUTPUT_FOLDER,
    VALIDATION_OUTPUTS,
)

torch.set_num_threads(25)


def save_validation_output(outputs: torch.Tensor, epoch: int, num_id: int):
    run = os.path.join(VALIDATION_OUTPUTS, str(num_id))
    os.makedirs(run, exist_ok=True)
    torch.save(outputs, os.path.join(run, str(epoch)))


def train_all():
    # For each CNN
    Ws_many = []
    bs_many = []

    losses_many = []
    accs_many = []

    points_to_collect = 5

    num_trajectory_record = 3
    collection_solution_after_epoch = 30
    batch_size = 128

    N_val = 500

    batch_size = 128

    N_val = 500
    dm = DatasetsManager()
    trainloader, valloader, testloader = dm.torch_load_cifar_10(
        batch_size=batch_size, validation_set=N_val
    )

    torch.save(trainloader, os.path.join(OUTPUT_FOLDER, "trainset.pth"))
    torch.save(valloader, VALIDATION_SET)
    torch.save(testloader, os.path.join(OUTPUT_FOLDER, "testset.pth"))

    # Collecting last epochs from each trajectory
    Ws_by_epochs_many = [[] for _ in range(points_to_collect)]
    bs_by_epochs_many = [[] for _ in range(points_to_collect)]

    # Collecting the whole trajectory
    Ws_trajectory = [[] for _ in range(num_trajectory_record)]
    bs_trajectory = [[] for _ in range(num_trajectory_record)]

    for point_id in range(points_to_collect):
        print("Optimization " + str(point_id))

        cnn = MediumCNN()

        criterion = nn.CrossEntropyLoss()
        lr = cnn.learning_rate
        optimizer = torch.optim.Adam(params=cnn.parameters(), lr=lr)

        for epoch in range(cnn.epochs):
            start_time = time.monotonic()
            epoch_number = epoch + 1
            correct = 0
            total_labels = 0
            # The authors halves every 10 epochs
            if epoch_number % 10 == 0:
                lr = lr / 2
                optimizer.param_groups[0]["lr"] = lr

            # TODO random choice
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if i % 100 == 0 or i == len(trainloader) - 1:
                    correct, total_labels = print_train_step(
                        epoch, i, labels, loss.item(), outputs, correct, total_labels
                    )

            # end epoch

            state_dict = deepcopy(cnn.state_dict())
            Ws_opt_out_now = [
                state_dict["conv1.weight"],
                state_dict["conv2.weight"],
                state_dict["conv3.weight"],
                state_dict["conv4.weight"],
                state_dict["fc1.weight"],
            ]

            bs_opt_out_now = [
                state_dict["conv1.bias"],
                state_dict["conv2.bias"],
                state_dict["conv3.bias"],
                state_dict["conv4.bias"],
                state_dict["fc1.bias"],
            ]

            outputs, val_acc, val_loss = cnn.get_validation_predictions(valloader)

            save_validation_output(outputs, epoch, point_id)
            print_end_epoch_step(
                epoch, loss.item(), correct * 100 / total_labels, val_loss, val_acc
            )
            if epoch >= collection_solution_after_epoch:
                Ws_by_epochs_many[point_id].append(Ws_opt_out_now)
                bs_by_epochs_many[point_id].append(bs_opt_out_now)
            if point_id < num_trajectory_record:
                Ws_trajectory[point_id].append(Ws_opt_out_now)
                bs_trajectory[point_id].append(bs_opt_out_now)

                # c  # lasses[point_id][epoch] =
            end_time = time.monotonic()
            print(f"Epoch training time: {end_time - start_time}")
        Ws_many.append(Ws_opt_out_now)
        bs_many.append(bs_opt_out_now)
        losses_many.append(val_loss)
        accs_many.append(val_acc)
        save_data(Ws_trajectory, WS_TRAJECTORY)
        save_data(bs_trajectory, BS_TRAJECTORY)
        save_data(Ws_many, WS_MANY)
        save_data(bs_many, BS_MANY)
        save_data(Ws_by_epochs_many, WS_BY_EPOCHS)
        save_data(bs_by_epochs_many, BS_BY_EPOCHS)

        torch.save(cnn.state_dict, f"saved_models/loss_landscapes/mediumCNN{point_id}")


if __name__ == "__main__":
    """
    Trains CNN ensembles needed to reproduce the following paper.

    Deep Ensembles: A Loss Landscape Perspective

    Original Code: https://github.com/deepmind/deepmind-research/tree/master/ensemble_loss_landscape
    """

    train_all()
