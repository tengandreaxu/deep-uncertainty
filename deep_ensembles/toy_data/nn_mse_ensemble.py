import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from deep_ensembles.models.mlp import MLP
from deep_ensembles.params.TrainingParameters import TrainingParameters
from deep_ensembles.toy_data.data_generator import generate_data

if __name__ == "__main__":

    xx, yy, x, y = generate_data(points=20, xrange=(-4, 4), std=3.0)
    mlps = []
    mlp_optimizers = []
    mlp_criterion = nn.MSELoss()
    M = 5
    for _ in range(M):
        net = MLP(hidden_layers=[100], activation="relu")  # standard MLP
        mlps.append(net)
        mlp_optimizers.append(
            torch.optim.Adam(
                params=net.parameters(), lr=TrainingParameters.learning_rate
            )
        )
    # train
    for i, net in enumerate(mlps):
        print("Training network ", i + 1)
        for epoch in range(TrainingParameters.epochs):
            mlp_optimizers[i].zero_grad()
            mlp_loss = mlp_criterion(yy, net(xx))
            if epoch == 0:
                print("initial loss: ", mlp_loss.item())
            mlp_loss.backward()
            mlp_optimizers[i].step()
        print("final loss: ", mlp_loss.item())

    plt.plot(xx.numpy(), yy.numpy(), "or", label="data points")
    plt.plot(x, y, "b", label="ground truth $y=x^3$")
    ys = []
    for net in mlps:
        ys.append(net(torch.tensor(x).float()).detach().numpy())
    ys = np.array(ys)
    mean = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    plt.plot(x, mean, label="MLP (MSE)", color="grey")
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
    plt.title("Figure 1 (left) from the paper")
    plt.legend()
    plt.savefig("deep_ensembles/plots/toy_data/mse_ensemble.png")
