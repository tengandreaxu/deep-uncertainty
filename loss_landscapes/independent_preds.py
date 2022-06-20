import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from loss_landscapes.func_utils import flatten, load_data
from loss_landscapes.models.MediumCNN import MediumCNN
from utils.linear_algebra import cosine_between
from utils.pytorch_custom import accuracy
from loss_landscapes.paths import WS_MANY, BS_MANY
from sklearn.manifold import TSNE
from matplotlib import patches as mpatch


if __name__ == "__main__":
    """
    Section 4.1 Figure 3 (a)

    Deep Ensembles: A Loss Landscape Perspective

    Original Code: https://github.com/deepmind/deepmind-research/tree/master/ensemble_loss_landscape
    """

    flat_p_list = []
    Ws_many = load_data(WS_MANY)
    bs_many = load_data(BS_MANY)
    points_to_collect = 5
    cos_matrix = np.zeros((points_to_collect, points_to_collect))
    for i in range(points_to_collect):
        flat_p_list.append(flatten(Ws_many[i], bs_many[i]))

    for i in range(points_to_collect):
        for j in range(i, points_to_collect):
            cos_matrix[i][j] = cosine_between(flat_p_list[i], flat_p_list[j])
            cos_matrix[j][i] = cos_matrix[i][j]

    plt.imshow(cos_matrix, cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    title = "Cosine Between Independent Solutions"
    plt.title(title)

    plt.xlabel("Independent Solution")
    plt.ylabel("Independent Solution")
    plt.savefig(f"plots/loss_landscapes/independent_results.png")
