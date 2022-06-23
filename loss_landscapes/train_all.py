import os
import argparse
from typing import Tuple
from loss_landscapes.FileNames import FileNames
import torch
import time
import numpy as np
import torch.nn as nn
import torch.utils.data as tdata
from copy import deepcopy
from helpers.DatasetsManager import DatasetsManager
from loss_landscapes.models.MediumCNN import MediumCNN
from loss_landscapes.models.SmallCNN import SmallCNN
from utils.pytorch_custom import print_end_epoch_step, print_train_step
from loss_landscapes.func_utils import save_data
from loss_landscapes.paths import (
    BS_BY_EPOCHS,
    BS_MANY,
    TEST_SET,
    VALIDATION_SET,
    WS_BY_EPOCHS,
    WS_MANY,
    WS_TRAJECTORY,
    BS_TRAJECTORY,
    OUTPUT_FOLDER,
    VALIDATION_OUTPUTS,
)

torch.set_num_threads(25)


def save_validation_output(
    outputs: torch.Tensor, output_dir: str, epoch: int, num_id: int
):
    if len(output_dir) > 0:
        out = os.path.join(output_dir, OUTPUT_FOLDER, FileNames.validation_outputs)
        os.makedirs(out, exist_ok=True)
    else:
        out = VALIDATION_OUTPUTS
    run = os.path.join(out, str(num_id))
    os.makedirs(run, exist_ok=True)
    torch.save(outputs, os.path.join(run, str(epoch)))


RANDOM_SAMPLE = "random_sampling"


def get_and_save_datasets(
    output_dir: str, N_val: int
) -> Tuple[
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """loads and saves the dataset for training, validation, and test"""
    dm = DatasetsManager()
    (
        trainloader,
        valloader,
        testloader,
        train_ds,
        val_ds,
        test_ds,
    ) = dm.torch_load_cifar_10(batch_size=1, validation_set=N_val)

    trainset_path = os.path.join(OUTPUT_FOLDER, FileNames.trainset)
    validationset_path = VALIDATION_SET
    testset_path = TEST_SET
    if len(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        new_output_folder = os.path.join(output_dir, OUTPUT_FOLDER)
        os.makedirs(new_output_folder, exist_ok=True)
        trainset_path = os.path.join(new_output_folder, FileNames.trainset)
        validationset_path = os.path.join(new_output_folder, FileNames.validationset)
        testset_path = os.path.join(new_output_folder, FileNames.testset)

    torch.save(trainloader, trainset_path)
    torch.save(valloader, validationset_path)
    torch.save(testloader, testset_path)

    return trainloader, valloader, testloader, train_ds, val_ds, test_ds


def train_all(model: str, independent_runs: int, output_dir: str, force_gpu: bool):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    if device == "cpu" and force_gpu:
        print("Something wrong, GPU not found")
        exit(1)

    # For each CNN
    Ws_many = []
    bs_many = []

    losses_many = []
    accs_many = []

    num_trajectory_record = 3
    collection_solution_after_epoch = 30
    batch_size = 128

    N_val = 500

    batch_size = 128

    trainloader, valloader, _, train_ds, _, _ = get_and_save_datasets(output_dir, N_val)

    # Collecting last epochs from each trajectory
    Ws_by_epochs_many = [[] for _ in range(independent_runs)]
    bs_by_epochs_many = [[] for _ in range(independent_runs)]

    # Collecting the whole trajectory
    Ws_trajectory = [[] for _ in range(num_trajectory_record)]
    bs_trajectory = [[] for _ in range(num_trajectory_record)]

    for point_id in range(independent_runs):
        print("Optimization " + str(point_id))
        if model == "smallCNN":
            cnn = SmallCNN()
        else:
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

            # random choice Batch training
            iterations = int(np.floor(float(len(trainloader)) / float(batch_size)))
            for i in range(iterations):
                train_sampler = tdata.RandomSampler(train_ds, num_samples=batch_size)
                dataloader = tdata.dataloader.DataLoader(
                    train_ds, batch_size=batch_size, sampler=train_sampler
                )
                for _, data in enumerate(dataloader, 0):

                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = cnn(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0 or i == len(trainloader) - 1:
                        correct, total_labels = print_train_step(
                            epoch,
                            i,
                            labels,
                            loss.item(),
                            outputs,
                            correct,
                            total_labels,
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

            save_validation_output(outputs, output_dir, epoch, point_id)
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

        if len(output_dir) > 0:
            ws_trajectory_folder = os.path.join(
                output_dir, OUTPUT_FOLDER, FileNames.ws_trajectory
            )
            bs_trajectory_folder = os.path.join(
                output_dir, OUTPUT_FOLDER, FileNames.bs_trajectory
            )

            ws_many_folder = os.path.join(output_dir, OUTPUT_FOLDER, FileNames.ws_many)
            bs_many_folder = os.path.join(output_dir, OUTPUT_FOLDER, FileNames.bs_many)

            ws_by_epochs_folder = os.path.join(
                output_dir, OUTPUT_FOLDER, FileNames.ws_by_epochs
            )
            bs_by_epochs_folder = os.path.join(
                output_dir, OUTPUT_FOLDER, FileNames.bs_by_epochs
            )
        else:
            ws_trajectory_folder = WS_TRAJECTORY
            bs_trajectory_folder = BS_TRAJECTORY

            ws_many_folder = WS_MANY
            bs_many_folder = BS_MANY

            ws_by_epochs_folder = WS_BY_EPOCHS
            bs_by_epochs_folder = BS_BY_EPOCHS
        save_data(Ws_trajectory, ws_trajectory_folder)
        save_data(bs_trajectory, bs_trajectory_folder)
        save_data(Ws_many, ws_many_folder)
        save_data(bs_many, bs_many_folder)
        save_data(Ws_by_epochs_many, ws_by_epochs_folder)
        save_data(bs_by_epochs_many, bs_by_epochs_folder)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(cnn.state_dict, os.path.join(output_dir, f"mediumCNN{point_id}"))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir", dest="output_dir", type=str, help="where to save all results"
    )
    parser.set_defaults(output_dir="")

    parser.add_argument(
        "--model", dest="model", type=str, help="whether smallCNN or mediumCNN"
    )
    parser.set_defaults(model="mediumCNN")

    parser.add_argument(
        "--independent-runs",
        dest="independent_runs",
        type=int,
        help="how many independent nets to train",
    )
    parser.set_defaults(independent_runs=5)

    parser.add_argument(
        "--force-gpu",
        dest="force_gpu",
        action="store_true",
        help="when running in a computer with GPU but this one doesn't get activate stop the script",
    )
    parser.set_defaults(force_gpu=False)
    return parser.parse_args()


if __name__ == "__main__":
    """
    Trains CNN ensembles needed to reproduce the following paper.

    Deep Ensembles: A Loss Landscape Perspective

    Original Code: https://github.com/deepmind/deepmind-research/tree/master/ensemble_loss_landscape
    """
    args = get_args()
    train_all(args.model, args.independent_runs, args.output_dir, args.force_gpu)
