import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from dataclasses import dataclass

import logging

logging.basicConfig(level=logging.INFO)


# *********************
# Plots Palette and Styles
# *********************
params = {
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}
pylab.rcParams.update(params)


@dataclass
class Plotter:
    def __init__(self):
        self.logger = logging.getLogger("Plotter")

    def plot_deep_ensemble_toy_data(
        self,
        noisy_points_x: np.array,
        nosy_points_y: np.array,
        ground_truth_x: np.array,
        ground_truth_y: np.array,
        prediction_mean: np.array,
        prediction_variance: np.array,
        label: str,
        title: str,
        file_name: str,
    ):
        std = np.sqrt(prediction_variance)

        self.logger.info("Average Std: {:.2f}".format(std.mean()))
        plt.plot(noisy_points_x, nosy_points_y, "or", label="data points")
        plt.plot(ground_truth_x, ground_truth_y, "b", label="ground truth $y=x^3$")
        plt.plot(ground_truth_x, prediction_mean, "grey", label=label)
        plt.fill_between(
            ground_truth_x.reshape(
                100,
            ),
            (prediction_mean - 3 * std).reshape(
                100,
            ),
            (prediction_mean + 3 * std).reshape(
                100,
            ),
            color="grey",
            alpha=0.3,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def scatter_plot(self, x: np.array, y: np.array, file_name: str):

        plt.scatter(x, y)
        plt.ylabel("Predicted Value")
        plt.xlabel("True Value")
        m, b = np.polyfit(x, y, 1)

        plt.plot(x, m * x + b, "r")
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
