import torch
import torch.nn as nn
from typing import Optional, Callable
from deep_ensembles.params.TrainingParameters import TrainingParameters


def train_mse_ensemble(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    inputs: int,
    num_models: Optional[int] = 5,
    hidden_layers: Optional[list] = [100],
    use_rmse: Optional[bool] = False,
):
    mlps = []
    for i in range(num_models):
        net = MLP(
            inputs=inputs, hidden_layers=hidden_layers, activation="relu"
        )  # standard MLP
        mlp_optimizer = torch.optim.Adam(
            params=net.parameters(), lr=TrainingParameters.learning_rate
        )

        print("Training network ", i + 1)
        for epoch in range(40):
            mlp_optimizer.zero_grad()

            mlp_loss = loss_fn(y, net(x.float()))

            if use_rmse:
                mlp_loss = torch.sqrt(mlp_loss)
            if epoch == 0:
                print("initial loss: ", mlp_loss.item())
            mlp_loss.backward()
            mlp_optimizer.step()
            print("current loss: ", mlp_loss.item())
        print("final loss: ", mlp_loss.item())
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

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation="relu"):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.nLayers = len(hidden_layers)
        self.net_structure = [inputs, *hidden_layers, outputs]
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
        return x


if __name__ == "__main__":
    mlp = MLP()
    breakpoint()
