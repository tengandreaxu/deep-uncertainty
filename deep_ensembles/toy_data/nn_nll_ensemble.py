import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from deep_ensembles.models.gm_mlp import GaussianMixtureMLP
from deep_ensembles.params.TrainingParameters import TrainingParameters
from deep_ensembles.toy_data.data_generator import generate_data
from losses.nll import NLLloss


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
    xx, yy, x, y = generate_data(points=20, xrange=(-4, 4), std=3.0)

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
        losses = train_gmm_step(gmm, gmm_optimizers, xx, yy)
        if epoch == 0:
            print("inital losses: ", losses)
    print("final losses: ", losses)

    means = []
    variances = []
    for i in range(gmm.num_models):
        model = getattr(gmm, "model_" + str(i))
        mean, var = model(torch.tensor(x).float())
        mean = mean.detach().numpy()
        var = var.detach().numpy()
        means.append(mean)
        variances.append(var)
        std = np.sqrt(var)
        plt.plot(x, mean, label="GMM (NLL) " + str(i + 1), alpha=0.5)
        plt.fill_between(
            x.reshape(
                100,
            ),
            (mean - std).reshape(
                100,
            ),
            (mean + std).reshape(
                100,
            ),
            alpha=0.1,
        )
    plt.plot(x, y, label="ground truth $y=x^3$", color="b")
    plt.plot(xx.numpy(), yy.numpy(), "or", label="data points")
    plt.title("Outputs of the network in the ensemble")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.close()

    mean, var = gmm(torch.tensor(x).float())
    mean = mean.detach().numpy()
    var = var.detach().numpy()
    std = np.sqrt(var)

    plt.fill_between(
        x.reshape(
            100,
        ),
        (mean - std).reshape(
            100,
        ),
        (mean + std).reshape(
            100,
        ),
        color="grey",
        alpha=0.3,
        label="$\sigma$",
    )
    plt.plot(x, mean, label="GMM mean (NLL)", color="grey")
    plt.plot(x, y, "b", label="ground truth $y=x^3$")
    plt.plot(xx.numpy(), yy.numpy(), "or", label="data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Figure 1 (right) from the paper")
    plt.legend()
    plt.savefig("deep_ensembles/plots/toy_data/nll_ensemble.png")
