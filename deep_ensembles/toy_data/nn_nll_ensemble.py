import torch
from deep_ensembles.models.gm_mlp import train_gmm_ensemble
from deep_ensembles.toy_data.data_generator import generate_data

from plotting.Plotter import Plotter

if __name__ == "__main__":
    plotter = Plotter()
    noisy_points_x, noisy_points_y, x, y = generate_data(
        points=20, xrange=(-4, 4), std=3.0
    )

    gmm, losses = train_gmm_ensemble(noisy_points_x, noisy_points_y, inputs=1)
    mean, var = gmm(torch.tensor(x).float())
    plotter.plot_deep_ensemble_toy_data(
        noisy_points_x=noisy_points_x.numpy(),
        nosy_points_y=noisy_points_y.numpy(),
        ground_truth_x=x,
        ground_truth_y=y,
        prediction_mean=mean.detach().numpy(),
        prediction_variance=var.detach().numpy(),
        label="GMLP NLL-Ensemble",
        title="Fourth Figure",
        file_name="deep_ensembles/plots/toy_data/nll_ensemble.png",
    )
