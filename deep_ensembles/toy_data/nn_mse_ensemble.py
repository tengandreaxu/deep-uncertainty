import torch
import torch.nn as nn
import numpy as np
from deep_ensembles.models.mlp import train_mse_ensemble
from deep_ensembles.toy_data.data_generator import generate_data
from plotting.Plotter import Plotter

if __name__ == "__main__":
    plotter = Plotter()
    noisy_points_x, noisy_points_y, x, y = generate_data(
        points=20, xrange=(-4, 4), std=3.0
    )

    mlps = train_mse_ensemble(noisy_points_x, noisy_points_y, nn.MSELoss(), 1)
    ys = []
    for net in mlps:
        ys.append(net(torch.tensor(x).float()).detach().numpy())
    ys = np.array(ys)
    mean = np.mean(ys, axis=0)
    var = np.std(ys, axis=0) ** 2

    plotter.plot_deep_ensemble_toy_data(
        noisy_points_x=noisy_points_x.numpy(),
        nosy_points_y=noisy_points_y.numpy(),
        ground_truth_x=x,
        ground_truth_y=y,
        prediction_mean=mean,
        prediction_variance=var,
        label="MLP MSE",
        title="First Figure",
        file_name="deep_ensembles/plots/toy_data/mse_ensemble.png",
    )
