import torch
import torch.nn as nn
from typing import Optional
from losses.nll import NLLloss
from deep_ensembles.models.gaussian_mlp import GaussianMLP
from deep_ensembles.params.TrainingParameters import TrainingParameters


def train_model_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """Training an individual gaussian MLP of the deep ensemble."""
    optimizer.zero_grad()
    mean, var = model(x.float())
    loss = NLLloss(y, mean, var)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_gmm_step(
    gmm: nn.Module,
    optimizers: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """Training the whole ensemble."""
    losses = []
    for i in range(gmm.num_models):
        model = getattr(gmm, "model_" + str(i))

        loss = train_model_step(model, optimizers[i], x, y)
        losses.append(loss)
    return losses


def train_gmm_ensemble(
    x: torch.Tensor,
    y: torch.Tensor,
    inputs: int,
    num_models: Optional[int] = 5,
    hidden_layers: Optional[list] = [100],
):
    """Launches the ensemble

    Args:
        x, predictors
        y, labels
        inputs, number of input units
        num_models, number of NNS
        hidden_layers, list of hidden layers
    """
    # Initialize A Gaussian Mixture Neural Net with num_models Gaussians
    gmm = GaussianMixtureMLP(
        inputs=inputs,
        num_models=num_models,
        hidden_layers=hidden_layers,
    )
    gmm_optimizers = []

    # Initialize As many optimizers as Gaussians
    for i in range(gmm.num_models):
        model = getattr(gmm, "model_" + str(i))
        gmm_optimizers.append(
            torch.optim.Adam(
                params=model.parameters(),
                lr=TrainingParameters.learning_rate,
                weight_decay=4e-5,
            )
        )

    # Train all Gaussian Models
    for epoch in range(TrainingParameters.epochs):
        losses = train_gmm_step(gmm, gmm_optimizers, x, y)
        if epoch == 0:
            print("inital losses: ", losses)
        print("current loss: ", losses)
    print("final losses: ", losses)
    return gmm, losses


class GaussianMixtureMLP(nn.Module):
    """Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(
        self,
        num_models=5,
        inputs=1,
        outputs=1,
        hidden_layers=[100],
        activation="relu",
    ):
        super(GaussianMixtureMLP, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.activation = activation

        # Sets the Gaussian Mixture Networks
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
