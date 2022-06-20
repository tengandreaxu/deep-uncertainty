from cgi import test
import os
import torch
import torchvision
import pandas as pd
import torchvision.transforms as transforms
from scipy.io import arff
from typing import Optional, Tuple
from torch.utils.data import random_split
from dataclasses import dataclass
from utils.paths import get_datasets_dir
import logging

logging.basicConfig(level=logging.INFO)
DATASETS = [
    "boston_housing",
    "concrete",
    "energy",
    "power_plant",
    "naval_plant",
    "kin8nm",
    "protein",
    "wine",
    "yacht",
    "year_msd",
]


@dataclass
class DatasetsManager:
    def __init__(self):
        self.path_datasets = get_datasets_dir()
        self.path_boston_housing = os.path.join(
            self.path_datasets, "boston_housing.csv"
        )
        self.path_concrete = os.path.join(self.path_datasets, "concrete_data.csv")
        self.path_energy_efficiency = os.path.join(
            self.path_datasets, "energy_efficiency.csv"
        )
        self.path_power_plant = os.path.join(
            self.path_datasets, "global_power_plant_database.csv"
        )
        self.path_kin8nm = os.path.join(self.path_datasets, "kin8nm.arff")
        self.path_naval_plant = os.path.join(
            self.path_datasets, "navalplantmaintenance.csv"
        )
        self.path_protein = os.path.join(self.path_datasets, "pdb_data_seq.csv")
        self.path_wine_quality = os.path.join(self.path_datasets, "winequality-red.csv")
        self.path_yacht = os.path.join(self.path_datasets, "yacht_hydro.csv")

        self.path_year_msd = os.path.join(self.path_datasets, "YearPredictionMSD.csv")
        self.logger = logging.getLogger("DatasetsManager")

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name in DATASETS:
            self.logger.info(dataset_name)

        # target: MEDV
        if dataset_name == "boston_housing":
            column_names = [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ]
            return pd.read_csv(self.path_boston_housing, names=column_names, sep="\s+")

        # target: concrete_compressive_strength
        elif dataset_name == "concrete":
            return pd.read_csv(self.path_concrete)

        # targets: Y1 Y2
        elif dataset_name == "energy":
            return pd.read_csv(self.path_energy_efficiency)

        # is this correct?
        # target: estimated_generation_gwh
        elif dataset_name == "power_plant":
            return pd.read_csv(self.path_power_plant)

        # target: y
        elif dataset_name == "kin8nm":
            return pd.DataFrame(arff.loadarff(self.path_kin8nm)[0])

        # target gt_c_decay, gt_t_decay
        elif dataset_name == "naval_plant":
            column_names = [
                "lever_position",
                "ship_speed",
                "gt_shaft",
                "gt_rate",
                "gg_rate",
                "sp_torque",
                "pp_torque",
                "hpt_temp",
                "gt_c_i_temp",
                "gt_c_o_temp",
                "hpt_pressure",
                "gt_c_i_pressure",
                "gt_c_o_pressure",
                "gt_exhaust_pressure",
                "turbine_inj_control",
                "fuel_flow",
                "gt_c_decay",
                "gt_t_decay",
            ]
            return pd.read_csv(self.path_naval_plant, sep="\s+", names=column_names)
        elif dataset_name == "protein":
            return pd.read_csv(self.path_protein)

        # target: quality
        elif dataset_name == "wine":
            return pd.read_csv(self.path_wine_quality)

        # target: Rr
        elif dataset_name == "yacht":
            return pd.read_csv(self.path_yacht)
        elif dataset_name == "year_msd":
            column_names = ["year"]
            columns = [str(i) for i in range(90)]
            column_names = column_names + columns
            return pd.read_csv(self.path_year_msd, names=column_names)
        else:
            raise Exception(
                f"Please specify one of the following datasets: {str(DATASETS)}"
            )

    def load_mnist(self) -> Tuple[list, list, list, list]:
        """returns the mnist dataset"""
        from mnist import MNIST

        mndata = MNIST("datasets/mnist")
        train_images, train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()
        return train_images, train_labels, test_images, test_labels

    def torch_load_cifar_10(
        self, batch_size: Optional[int] = 4, validation_set: Optional[int] = 0
    ) -> Tuple[
        torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader
    ]:
        """Returns CIFAR-10 trainset and testset"""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        if validation_set > 0:
            train_ds, val_ds = random_split(
                trainset, [len(trainset) - validation_set, validation_set]
            )
            valloader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=2
            )
        trainloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return trainloader, valloader, testloader

    def get_cifar_10_label_names(self) -> list:
        return [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]


if __name__ == "__main__":
    dm = DatasetsManager()
    dm.torch_load_cifar_10()
