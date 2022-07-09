import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
from deep_ensembles.params.TrainingParameters import TrainingParameters
from sklearn.preprocessing import StandardScaler


def train_mse_ensemble(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    inputs: int,
    num_models: Optional[int] = 5,
    hidden_layers: Optional[list] = [100],
    use_rmse: Optional[bool] = False,
    y_scaler: Optional[StandardScaler] = None,
):
    """trains an ensemble of networks"""
    mlps = []

    # Train all models
    for i in range(num_models):

        # Instantiate MLP
        net = MLP(
            inputs=inputs,
            hidden_layers=hidden_layers,
            activation="relu",
            y_scaler=y_scaler,
        )
        mlp_optimizer = torch.optim.Adam(
            params=net.parameters(), lr=TrainingParameters.learning_rate
        )

        print("Training network ", i + 1)
        x.requires_grad = True
        # Train
        for epoch in range(40):
            mlp_optimizer.zero_grad()

            # l(\theta, x, y)
            mlp_loss = loss_fn(y.float(), net(x.float()))

            mse_loss = mlp_loss.item()
            if use_rmse:
                mse_loss = np.sqrt(mse_loss)
            if epoch == 0:
                print("initial loss: ", mse_loss)
            mlp_loss.backward(retain_graph=True)

            # Adversatial Perturbation
            gradient = x.grad.sign()
            perturbed_data = x + 0.01 * gradient
            output = net(perturbed_data.float())
            mlp_loss += loss_fn(y.float(), output)
            net.zero_grad()
            mlp_loss.backward()

            mlp_optimizer.step()
            print("current loss: ", mse_loss)
        print("final loss: ", mse_loss)
        mlps.append(net)
    return mlps


class MLP(nn.Module):
    """Multilayer perceptron (MLP) with tanh/sigmoid activation functions
    implemented in PyTorch for regression tasks.

    Attributes:
        inputs (int): inputs of the network
        outputs (int): outputs of the network
        hidden_layers (list): layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
        activation (string): activation function used ('relu', 'tanh' or 'sigmoid')

    """

    def __init__(
        self,
        inputs=1,
        outputs=1,
        hidden_layers=[100],
        activation="relu",
        y_scaler: Optional[StandardScaler] = None,
    ):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.nLayers = len(hidden_layers)
        self.net_structure = [inputs, *hidden_layers, outputs]
        self.y_scaler = y_scaler
        if activation == "relu":
            self.act = torch.relu
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        else:
            assert 'Use "relu","tanh" or "sigmoid" as activation.'
        # create linear layers y = Wx + b

        for i in range(self.nLayers + 1):
            setattr(
                self,
                "layer_" + str(i),
                nn.Linear(self.net_structure[i], self.net_structure[i + 1]),
            )

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, "layer_" + str(i))
            x = self.act(layer(x))
        layer = getattr(self, "layer_" + str(self.nLayers))
        x = layer(x)

        # transform scaler
        # x = torch.Tensor(
        #     x.detach().numpy() * self.y_scaler.scale_ + self.y_scaler.mean_
        # ).requires_grad_(True)

        return x


if __name__ == "__main__":
    mlp = MLP()
    breakpoint()
