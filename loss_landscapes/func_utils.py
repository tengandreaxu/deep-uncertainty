import pickle
import numpy as np
import torch
from typing import Any, Optional
from utils.scores import brier_scores
from loss_landscapes.models.SmallCNN import SmallCNN
from loss_landscapes.models.MediumCNN import MediumCNN
from loss_landscapes.ModelNames import ModelNames


def get_cnn(model_name: str) -> torch.nn.Module:
    """returns a cnn model from the name"""
    if model_name not in [ModelNames.mediumCNN, ModelNames.smallCNN]:
        raise Exception("Please specify a valid model: [smallCNN, mediumCNN]")

    if model_name == ModelNames.mediumCNN:
        return MediumCNN()
    else:
        return SmallCNN()


def flatten(Ws1: np.array, bs1: np.array):
    lists_now = []

    for W_now in Ws1:
        lists_now.append(W_now.reshape([-1]))

    for b_now in bs1:
        lists_now.append(b_now.reshape([-1]))
    return np.concatenate(lists_now, axis=0)


def reform(flat1: np.ndarray, Ws: torch.Tensor, bs: torch.Tensor):
    """reforms weight and bias coefficients from the subspace sample"""
    sofar = 0
    Ws_out_now = []
    bs_out_now = []
    for W in Ws:
        shape_now = list(W.shape)
        size_now = np.prod(shape_now)
        elements = flat1[sofar : sofar + size_now]
        sofar = sofar + size_now
        Ws_out_now.append(np.array(elements).reshape(shape_now))
    for b in bs:
        shape_now = list(b.shape)
        size_now = np.prod(shape_now)
        elements = flat1[sofar : sofar + size_now]
        sofar = sofar + size_now
        bs_out_now.append(np.array(elements).reshape(shape_now))
    return Ws_out_now, bs_out_now


def save_data(data: Any, name: str):
    with open(name, "wb") as f:
        pickle.dump(data, f)
        print(f"Saved: {name}")


def load_data(file_name: str):

    try:
        with open(file_name, "rb") as f:
            x = pickle.load(f)
    except:
        Exception(f"File not Found {file_name}")
    return x


def average_var(w_list: list) -> list:
    """Average a list of weights trained in different epochs"""
    avg = [[] for _ in w_list[0]]
    for w_now in w_list:
        for i, w in enumerate(w_now):
            avg[i].append(w)

    for i, v in enumerate(avg):

        avg[i] = np.mean(np.stack(v, axis=0), axis=0)

    return avg


def choose_k_from_n(n, k):
    """Returns a list of all possible k-subset from [1, ..., n]
    Don't scale well for large n. Use with caution."""
    if k > n or k < 1 or n < 1:
        return []
    if k == n:
        return [list(range(1, n + 1))]
    if k == 1:
        return [[i] for i in range(1, n + 1)]
    a = choose_k_from_n(n - 1, k)
    b = choose_k_from_n(n - 1, k - 1)
    b_new = []
    for g in b:
        b_new.append(g + [n])
    return a + b_new


def brier_scores(y: np.ndarray, probs: np.ndarray) -> float:
    """Computes Brier's score for multicategorical labels

    Args:
        y, the labels
        predicted probs

    Returns:
        the brier score
    """
    y_hot = np.zeros((y.size, y.max() + 1))
    y_hot[np.arange(y.size), y] = 1

    return np.mean(np.sum((probs - y_hot) ** 2, axis=1))


def get_acc_brier(y: np.ndarray, pred: torch.Tensor):
    if isinstance(pred, np.ndarray):
        pred = torch.Tensor(pred)
    preds = torch.max(pred, 1).indices.numpy()
    acc = np.mean(preds == y)

    pred_probs = torch.softmax(pred, 1).numpy()
    brier = brier_scores(y, pred_probs)  # brier_scores(y, probs=pred)

    return acc, brier


def get_all_models_metrics(
    pred_list: list,
    y_test: np.ndarray,
    max_ens_size: Optional[int] = 5,
    ens_size_list=None,
):
    """Given a list of model predictions, compute the accuracy and
    brier score for each individual model as well as the ensemble of them."""
    acc_list = []
    acc_list_ensemble = []
    b_list = []
    b_list_ensemble = []
    num_models = len(pred_list)
    for i in range(num_models):
        acc, brier = get_acc_brier(y_test, pred_list[i])
        acc_list.append(acc)
        b_list.append(brier)

    max_ens_size = np.min([max_ens_size, num_models])
    if ens_size_list is None:
        ens_size_list = range(1, max_ens_size + 1)
    for ens_size in ens_size_list:
        # Pick all possible subset with size of ens_size from available models.
        # Compute ensemble for each such subset.
        ens_index_list = choose_k_from_n(num_models, ens_size)
        ens_acc = []
        ens_brier = []
        for ens_ind in ens_index_list:
            ens_pred_list = []
            for ind in ens_ind:
                ens_pred_list.append(pred_list[ind - 1])
            acc, brier = get_acc_brier(y_test, ens_pred_list[0])
            ens_acc.append(acc)
            ens_brier.append(brier)
        acc_list_ensemble.append(np.mean(ens_acc))
        b_list_ensemble.append(np.mean(ens_brier))
    metrics = {
        "accuracy": {},
        "brier": {},
    }
    metrics["accuracy"]["individual"] = acc_list
    metrics["accuracy"]["ensemble"] = acc_list_ensemble
    metrics["brier"]["individual"] = b_list
    metrics["brier"]["ensemble"] = b_list_ensemble
    return metrics
