import numpy as np


def get_gaussian_sample(var_mean, var_std, scale=1.0):
    var_sample = np.random.normal(var_mean, scale * var_std)
    return var_sample


def get_pca_gaussian_flat_sampling(pca, means, rank, scale=1.0):
    """low-rank gaussian subspace"""
    standard_normals = np.random.normal(loc=0.0, scale=scale, size=(rank))
    shifts = pca.inverse_transform(standard_normals)
    return shifts + means


def get_rand_norm_direction(shape):
    random_direction = np.random.normal(loc=0.0, scale=1.0, size=shape)
    random_direction_normed = random_direction / np.linalg.norm(random_direction)
    return random_direction_normed
