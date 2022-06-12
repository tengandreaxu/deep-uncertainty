import torch


def NLLloss(y, mean, var):
    """Negative log-likelihood loss function."""
    return (torch.log(var) + ((y - mean).pow(2)) / var).sum()
