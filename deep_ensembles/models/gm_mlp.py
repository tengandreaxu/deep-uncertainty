import torch
import torch.nn as nn
from deep_ensembles.models.gaussian_mlp import GaussianMLP


class GaussianMixtureMLP(nn.Module):
    """Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(
        self, num_models=5, inputs=1, outputs=1, hidden_layers=[100], activation="relu"
    ):
        super(GaussianMixtureMLP, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.activation = activation
        for i in range(self.num_models):
            model = GaussianMLP(
                inputs=self.inputs,
                outputs=self.outputs,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
            )
            setattr(self, "model_" + str(i), model)

    def forward(self, x):
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, "model_" + str(i))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
        return mean, variance
