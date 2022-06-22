import torch
import numpy as np


def brier_scores(labels: torch.Tensor, probs: torch.Tensor):
    """Compute elementwise Brier score.

    Args:
      labels: Tensor of integer labels shape [N1, N2, ...]
      probs: Tensor of categorical probabilities of shape [N1, N2, ..., M].
      logits: If `probs` is None, class probabilities are computed as a softmax
        over these logits, otherwise, this argument is ignored.

    Returns:
      Tensor of shape [N1, N2, ...] consisting of Brier score contribution from
      each element. The full-dataset Brier score is an average of these values.
    """

    nlabels = probs.shape[-1]
    flat_probs = probs.reshape([-1, nlabels])
    flat_labels = labels.reshape([len(flat_probs)])

    plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]
    out = np.square(flat_probs).sum(axis=-1) - 2 * plabel
    return out.reshape(labels.shape) + 1
