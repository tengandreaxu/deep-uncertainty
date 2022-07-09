import torch
from deep_ensembles.models.mlp import MLP
import torch.nn.functional as F


class GaussianMLP(MLP):
    """Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(
        self,
        inputs=1,
        outputs=1,
        hidden_layers=[100],
        activation="relu",
    ):
        super(GaussianMLP, self).__init__(
            inputs=inputs,
            outputs=2 * outputs,
            hidden_layers=hidden_layers,
            activation=activation,
        )
        self.inputs = inputs
        self.outputs = outputs

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, "layer_" + str(i))
            x = self.act(layer(x))

        # get the last layer
        layer = getattr(self, "layer_" + str(self.nLayers))
        x = layer(x)
        mean, variance = torch.split(x, self.outputs, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance
