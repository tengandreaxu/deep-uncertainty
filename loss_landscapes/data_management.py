import os
import torch
from loss_landscapes.func_utils import load_data
from loss_landscapes.FileNames import FileNames


def load_saved_test_set(model_folder: str):
    return torch.load(os.path.join(model_folder, FileNames.testset))


def load_saved_validation_set(model_folder: str):
    return torch.load(os.path.join(model_folder, FileNames.validationset))


def load_saved_validation_set_labels(model_folder: str) -> torch.Tensor:
    validator = torch.load(os.path.join(model_folder, FileNames.validationset))
    vals = []
    for valid in validator:
        _, labels = valid
        vals.append(labels)
    vals = torch.cat(vals, 0)
    return vals


def load_validation_output(model_folder: str, id: int, epoch: int):
    return torch.load(
        os.path.join(model_folder, FileNames.validation_outputs, str(id), str(epoch))
    )

def load_saved_model(model_folder: str, id: int)-> torch.nn.Module:
    return torch.load(os.path.join(model_folder, f"CNN{id}"))