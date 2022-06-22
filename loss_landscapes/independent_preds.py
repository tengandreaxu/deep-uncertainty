import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from loss_landscapes.func_utils import flatten, load_data
from utils.linear_algebra import cosine_between
from loss_landscapes.paths import VALIDATION_SET, WS_MANY, BS_MANY, VALIDATION_OUTPUTS


def get_indepent_preds_test(
    points_to_collect: Optional[int] = 5,
    N_val: Optional[int] = 500,
    classes: Optional[int] = 10,
    epochs: Optional[int] = 40,
) -> np.array:
    """returns a numpy array of independent runs predictions"""
    independent_preds_test = np.zeros((points_to_collect, N_val, classes))
    for id_now in range(points_to_collect):

        validation_output = torch.load(
            os.path.join(VALIDATION_OUTPUTS, str(id_now), str(epochs - 1))
        )
        independent_preds_test[id_now] = validation_output.detach().numpy()
    return independent_preds_test


def get_validation_labels() -> torch.Tensor:
    """returns a Tensor of the validation set labels"""
    vals = []
    validator = torch.load(VALIDATION_SET)
    for valid in validator:
        _, labels = valid
        vals.append(labels)
    return torch.cat(vals, 0)


def plot_independent_runs_cosine():
    """Plots Figure 3(b)"""
    flat_p_list = []

    Ws_many = load_data(WS_MANY)
    bs_many = load_data(BS_MANY)
    points_to_collect = 5
    cos_matrix = np.zeros((points_to_collect, points_to_collect))
    for i in range(points_to_collect):
        flat_p_list.append(flatten(Ws_many[i], bs_many[i]))

    for i in range(points_to_collect):
        for j in range(i, points_to_collect):
            cos_matrix[i][j] = cosine_between(flat_p_list[i], flat_p_list[j])
            cos_matrix[j][i] = cos_matrix[i][j]

    plt.imshow(cos_matrix, cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    title = "Cosine Between Independent Solutions"
    plt.title(title)

    plt.xlabel("Independent Solution")
    plt.ylabel("Independent Solution")
    plt.savefig(f"plots/loss_landscapes/independent_results.png")


def plot_independent_runs_predictions():
    """plots Figure 3(a)"""
    preds_now = get_indepent_preds_test()
    targets_now = get_validation_labels()
    points_to_collect = 5
    classes_predicted = np.argmax(preds_now, axis=-1)
    fractional_differences = np.mean(
        classes_predicted.reshape([1, points_to_collect, len(targets_now)])
        != classes_predicted.reshape([points_to_collect, 1, len(targets_now)]),
        axis=-1,
    )

    plt.imshow(
        fractional_differences, interpolation="nearest", cmap="bwr", origin="lower"
    )
    plt.colorbar()
    plt.grid("off")

    title = "Disagreement Fraction Btw Independent Solutions"
    plt.title(title)

    plt.xlabel("Independent Solution")
    plt.ylabel("Independent Solution")
    plt.savefig(f"plots/loss_landscapes/disagreement_in_independent_solutions.png")


if __name__ == "__main__":
    """
    Section 4.1 Figure 3 (a)

    Deep Ensembles: A Loss Landscape Perspective

    Original Code: https://github.com/deepmind/deepmind-research/tree/master/ensemble_loss_landscape
    """

    plot_independent_runs_cosine()
    plot_independent_runs_predictions()
