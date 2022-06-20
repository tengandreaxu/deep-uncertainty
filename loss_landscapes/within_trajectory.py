import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from loss_landscapes.func_utils import flatten, load_data
from loss_landscapes.models.MediumCNN import MediumCNN
from utils.linear_algebra import cosine_between
from utils.pytorch_custom import accuracy
from loss_landscapes.paths import (
    WS_TRAJECTORY,
    BS_TRAJECTORY,
    VALIDATION_OUTPUTS,
    VALIDATION_SET,
)
from sklearn.manifold import TSNE
from matplotlib import patches as mpatch

# Plot Style
mpl.style.use("seaborn-colorblind")
mpl.rcParams.update(
    {"font.size": 14, "lines.linewidth": 2, "figure.figsize": (6, 6 / 1.61)}
)
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.marker"] = None
mpl.rcParams["axes.grid"] = True

DEFAULT_FONTSIZE = 13
mpl.rcParams.update(
    {
        "font.size": DEFAULT_FONTSIZE,
        "lines.linewidth": 2,
        "legend.fontsize": DEFAULT_FONTSIZE,
        "axes.labelsize": DEFAULT_FONTSIZE,
        "xtick.labelsize": DEFAULT_FONTSIZE,
        "ytick.labelsize": DEFAULT_FONTSIZE,
        "figure.figsize": (7, 7.0 / 1.4),
    }
)


def plot_t_sne_distribution(
    trajectory_preds_test, epochs: int, num_trajectory_record: Optional[int] = 3
):
    reshaped_prediction = trajectory_preds_test.reshape([-1, 500 * 10])

    prediction_embed = TSNE(n_components=2).fit_transform(reshaped_prediction)
    traj_embed = prediction_embed.reshape([num_trajectory_record, epochs, 2])

    colors_list = ["r", "b", "g"]
    labels_list = ["traj_{}".format(i) for i in range(num_trajectory_record)]
    for i in range(num_trajectory_record):
        plt.plot(
            traj_embed[i, :, 0],
            traj_embed[i, :, 1],
            color=colors_list[i],
            alpha=0.8,
            linestyle="",
            marker="o",
            label=labels_list[i],
        )
        plt.plot(
            traj_embed[i, :, 0],
            traj_embed[i, :, 1],
            color=colors_list[i],
            alpha=0.3,
            linestyle="-",
            marker="",
        )
        plt.plot(
            traj_embed[i, 0, 0],
            traj_embed[i, 0, 1],
            color=colors_list[i],
            alpha=1.0,
            linestyle="",
            marker="*",
            markersize=20,
        )
        plt.legend(loc=1)
        plt.savefig("plots/loss_landscapes/t_sne_independent.png")


def get_trajectory_preds_test(
    Ws_trajectory: list,
    bs_trajectory: list,
    load_saved_validation: Optional[bool] = False,
    num_trajectory_record: Optional[int] = 3,
    N_val: Optional[int] = 500,
    classes: Optional[int] = 10,
) -> list:
    """Return a list of prediction for each independent training for each epoch"""
    epochs = len(Ws_trajectory[0])
    trajectory_preds_test = np.zeros((num_trajectory_record, epochs, N_val, classes))
    validator = torch.load(VALIDATION_SET)
    if load_saved_validation:
        for id_now in range(num_trajectory_record):
            for e in range(epochs):
                validation_output = torch.load(
                    os.path.join(VALIDATION_OUTPUTS, str(id_now), str(e))
                )
                vals = []
                for valid in validator:
                    _, labels = valid
                    vals.append(labels)
                vals = torch.cat(vals, 0)

                acc = accuracy(validation_output, vals)
                print(f"Run: {id_now} Epoch: {e} Val Accuracy: {acc:.3f}")

                trajectory_preds_test[id_now][e] = validation_output.detach().numpy()
    else:
        with torch.no_grad():
            for id_now in range(num_trajectory_record):

                for e in range(epochs):
                    state_dict = {}
                    cnn = MediumCNN()
                    for j, W in enumerate(Ws_trajectory[id_now][e]):
                        if j == 4:
                            state_dict[f"fc1.weight"] = W
                        else:
                            level = j + 1
                            state_dict[f"conv{level}.weight"] = W

                    for j, b in enumerate(bs_trajectory[id_now][e]):
                        if j == 4:
                            state_dict[f"fc1.bias"] = b
                        else:
                            level = j + 1
                            state_dict[f"conv{level}.bias"] = b

                    cnn.load_state_dict(state_dict)
                    val_output, acc, loss = cnn.get_validation_predictions(validator)
                    print(
                        f"Run: {id_now} Epoch: {e} Val Accuracy: {acc:.3f} Loss: {loss:.3f}"
                    )

                    trajectory_preds_test[id_now][e] = val_output.detach().numpy()
    return trajectory_preds_test


def plot_trajectory_preds(trajectory_preds_test: list):

    epochs = len(trajectory_preds_test)
    len_validation = len(trajectory_preds_test[0])

    classes_predicted = np.argmax(trajectory_preds_test, axis=-1)
    fractional_differences = np.mean(
        classes_predicted.reshape([1, epochs, len_validation])
        != classes_predicted.reshape([epochs, 1, len_validation]),
        axis=-1,
    )

    plt.imshow(
        fractional_differences, interpolation="nearest", cmap="bwr", origin="lower"
    )
    plt.colorbar()
    plt.grid("off")

    title = "Disagreement Fraction Along Train Trajectory"
    plt.title(title)

    plt.xlabel("Checkpoint id")
    plt.ylabel("Checkpoint id")
    plt.savefig("plots/loss_landscapes/disagreement_of_predictions.png")
    plt.close()


def plot_trajectory(Ws_trajectory: list, bs_trajectory: list):
    num_epochs = len(Ws_trajectory)
    cos_matrix = np.zeros((num_epochs, num_epochs))

    flat_p_list = []
    for e in range(num_epochs):
        flat_p_list.append(flatten(Ws_trajectory[e], bs_trajectory[e]))

    for i in range(num_epochs):
        for j in range(i, num_epochs):
            cos_matrix[i][j] = cosine_between(flat_p_list[i], flat_p_list[j])
            cos_matrix[j][i] = cos_matrix[i][j]
    plt.imshow(cos_matrix, interpolation="nearest", cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    title = "Cosine Along Training Trajectory"
    plt.title(title)

    plt.xlabel("Checkpoint ID")
    plt.ylabel("Checkpoint ID")
    plt.savefig("plots/loss_landscapes/cosing_along_train_trajectory.png")
    plt.close()


if __name__ == "__main__":
    """
    Section 4.1 Figure 2 (a), Figure 2(b), Figure 2(c)

    Deep Ensembles: A Loss Landscape Perspective

    Original Code: https://github.com/deepmind/deepmind-research/tree/master/ensemble_loss_landscape
    """

    Ws_trajectory = load_data(WS_TRAJECTORY)
    bs_trajectory = load_data(BS_TRAJECTORY)

    plot_trajectory(Ws_trajectory=Ws_trajectory[0], bs_trajectory=bs_trajectory[0])

    trajectory_preds_test = get_trajectory_preds_test(
        Ws_trajectory, bs_trajectory, load_saved_validation=True
    )
    epochs = len(trajectory_preds_test[0])

    plot_trajectory_preds(trajectory_preds_test[0])

    plot_t_sne_distribution(trajectory_preds_test, epochs)
