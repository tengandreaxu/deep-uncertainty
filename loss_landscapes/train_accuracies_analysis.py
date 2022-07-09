import argparse
from loss_landscapes.data_management import (
    load_saved_test_set,
    load_saved_validation_set,
    load_saved_validation_set_labels,
)
from loss_landscapes.models import MediumCNN, SmallCNN
from loss_landscapes.ModelNames import ModelNames
from loss_landscapes.func_utils import get_cnn
from loss_landscapes.data_management import load_saved_model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        help="whether we are loading a mediumCNN or a smallCNN",
        required=True,
    )

    parser.add_argument(
        "--model-folder",
        dest="model_folder",
        type=str,
        help="where the model results are stored",
        required=True,
    )

    parser.add_argument(
        "--independent-runs",
        dest="independent_runs",
        type=int,
        help="how many independent runs we'vs tested",
        required=True,
    )
    args = parser.parse_args()

    if args.model not in [ModelNames.mediumCNN, ModelNames.smallCNN]:
        raise Exception(f"Please specify a valid CNN: [smallCNN, mediumCNN]")

    return args


def analyze_accuracy_mean_and_std(model: str, model_folder: str, independent_runs: int):
    """computes mean and std on validation and testing set"""

    validation_set = load_saved_validation_set(model_folder)
    test_set = load_saved_test_set(model_folder)

    val_acc = []
    test_acc = []
    for i in range(independent_runs):
        state_dict = load_saved_model(model_folder, i)
        cnn = get_cnn(model)
        cnn.load_state_dict(state_dict)
        breakpoint()
        _, val_acc, _ = cnn.get_validation_predictions(validation_set)


if __name__ == "__main__":
    """
    Understanding if our torch implementation differs from the tensorflow v1
    version given by the authors
    """

    args = get_args()
    analyze_accuracy_mean_and_std(args.model, args.model_folder, args.independent_runs)
