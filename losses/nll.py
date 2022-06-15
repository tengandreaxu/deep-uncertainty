import torch


def NLLloss(y, mean, var):
    """Negative log-likelihood loss function."""
    return ((torch.log(var) / 2) + ((y - mean) ** 2) / (2 * var)).sum()
