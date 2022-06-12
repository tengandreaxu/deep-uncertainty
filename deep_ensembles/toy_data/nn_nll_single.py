import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from deep_ensembles.models.gaussian_mlp import GaussianMLP
from deep_ensembles.params.TrainingParameters import TrainingParameters
from deep_ensembles.toy_data.data_generator import generate_data
from losses.nll import NLLloss

if __name__ == "__main__":
    xx, yy, x, y = generate_data(points=20, xrange=(-4, 4), std=3.0)
    gmlp = GaussianMLP(hidden_layers=[100])
    gmlp_optimizer = torch.optim.Adam(
        params=gmlp.parameters(), lr=TrainingParameters.learning_rate
    )

    for epoch in range(TrainingParameters.epochs * 3):
        gmlp_optimizer.zero_grad()
        mean, var = gmlp(xx)
        gmlp_loss = NLLloss(yy, mean, var)  # NLL loss
        # gmlp_loss = (yy - mean).pow(2).mean() # MSE loss for testing
        if epoch == 0:
            print("initial loss: ", gmlp_loss.item())
        gmlp_loss.backward()
        gmlp_optimizer.step()
    print("final loss: ", gmlp_loss.item())

    plt.plot(xx.numpy(), yy.numpy(), "or", label="data points")
    plt.plot(x, y, "b", label="ground truth $y=x^3$")
    mean, var = gmlp(torch.tensor(x).float())
    mean = mean.detach().numpy()
    var = var.detach().numpy()
    std = np.sqrt(var)
    plt.plot(x, mean, "grey", label="GMLP (NLL)")
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
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Figure 1 (second from left) from the paper")
    plt.legend()
    plt.savefig("deep_ensembles/plots/toy_data/nll_single.png")
