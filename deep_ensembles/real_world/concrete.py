import argparse
import torch
import os
import torch.nn as nn
import numpy as np
from deep_ensembles.models.gm_mlp import train_gmm_ensemble
from deep_ensembles.models.mlp import train_mse_ensemble
from deep_ensembles.params.TrainingParameters import TrainingParameters
from helpers.DatasetsManager import DatasetsManager
from plotting.Plotter import Plotter
from utils.data_manipulation import torch_tensor_train_test_split
from losses.nll import NLLloss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss", dest="loss", type=str)
    parser.set_defaults(loss="mse")

    args = parser.parse_args()
    dm = DatasetsManager()
    plotter = Plotter()
    df = dm.load_dataset("concrete")

    y = df.pop("concrete_compressive_strength")

    losses = []
    inputs = len(df.columns)
    scatter_plot_name = "deep_ensembles/plots/real_world"
    hidden_layers = [50]
    for fold in range(TrainingParameters.folds):

        (
            X_train,
            X_test,
            y_train,
            y_test,
            x_scaler,
            y_scaler,
        ) = torch_tensor_train_test_split(df, y, shuffle=True, standardize=True)

        if args.loss == "mse":
            mlps = train_mse_ensemble(
                X_train,
                y_train,
                nn.MSELoss(),
                inputs=inputs,
                use_rmse=True,
                hidden_layers=hidden_layers,
                y_scaler=y_scaler,
            )

            # Taking all predictions
            ys = []
            for net in mlps:
                ys.append(net(torch.tensor(X_test).float()).detach().numpy())

            ys = np.array(ys)
            mean = np.mean(ys, axis=0)
            var = np.std(ys, axis=0) ** 2

            predicted_y = torch.Tensor(mean).reshape(len(mean))

            plotter.scatter_plot(
                y_test.reshape(1, y_test.shape[0]).numpy()[0],
                predicted_y.numpy(),
                file_name=os.path.join(scatter_plot_name, "concrete_mse.png"),
            )
            loss = np.sqrt(nn.MSELoss()(y_test, predicted_y.reshape(-1, 1)))
            losses.append(loss)

        else:
            gmm, losses = train_gmm_ensemble(
                X_train, y_train, inputs=inputs, hidden_layers=hidden_layers
            )

            mean, var = gmm(X_test.float())
            avg_loss = NLLloss(y_test, mean, var)
            plotter.scatter_plot(
                y_test.numpy().reshape(1, y_test.shape[0])[0],
                mean.detach().numpy().reshape(1, mean.shape[0])[0],
                file_name=os.path.join(scatter_plot_name, "concrete_nll.png"),
            )
            loss = avg_loss.detach().numpy()

            losses.append(loss)

    avg = np.mean(losses)
    std = np.std(losses)
    print(f"Loss: \t {round(avg, 3)} +- {round(std, 3)}")
