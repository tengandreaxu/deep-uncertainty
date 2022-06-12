import torch
import numpy as np


def generate_data(points=20, xrange=(-4, 4), std=3.0):
    """Generates ground truth and noisy data"""
    # noisy data
    xx = torch.tensor([[np.random.uniform(*xrange)] for i in range(points)])
    yy = torch.tensor([[x**3 + np.random.normal(0, std)] for x in xx])

    # ground truth
    x = np.linspace(-6, 6, 100).reshape(100, 1)
    y = x**3
    return xx, yy, x, y


if __name__ == "__main__":
    xx, yy = generate_data(
        points=20, xrange=(-4, 4), std=3.0
    )  # generate data set of 20 samples
