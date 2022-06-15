import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from deep_ensembles.models.gm_mlp import GaussianMixtureMLP
from deep_ensembles.params.TrainingParameters import TrainingParameters
from deep_ensembles.toy_data.data_generator import generate_data
from losses.nll import NLLloss
from plotting.Plotter import Plotter


def train_model_step(model, optimizer, x, y):
    """Training an individual gaussian MLP of the deep ensemble."""
    optimizer.zero_grad()
    mean, var = model(x)
    loss = NLLloss(y, mean, var)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_gmm_step(gmm, optimizers, x, y):
    """Training the whole ensemble."""
    losses = []
    for i in range(gmm.num_models):
        model = getattr(gmm, "model_" + str(i))
        loss = train_model_step(model, optimizers[i], x, y)
        losses.append(loss)
    return losses


if __name__ == "__main__":
    plotter = Plotter()
    noisy_points_x, noisy_points_y, x, y = generate_data(
        points=20, xrange=(-4, 4), std=3.0
    )

    gmm = GaussianMixtureMLP(num_models=5, hidden_layers=[100])
    gmm_optimizers = []

    for i in range(gmm.num_models):
        model = getattr(gmm, "model_" + str(i))
        gmm_optimizers.append(
            torch.optim.Adam(
                params=model.parameters(),
                lr=TrainingParameters.learning_rate,
                weight_decay=4e-5,
            )
        )
    for epoch in range(TrainingParameters.epochs):
        losses = train_gmm_step(gmm, gmm_optimizers, noisy_points_x, noisy_points_y)
        if epoch == 0:
            print("inital losses: ", losses)
    print("final losses: ", losses)

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
