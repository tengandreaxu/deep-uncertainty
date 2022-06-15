import os
import pandas as pd
from scipy.io import arff
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


if __name__ == "__main__":
    dm = DatasetsManager()
    for data in DATASETS:
        print(data)
        df = dm.load_dataset("year_msd")
        breakpoint()
