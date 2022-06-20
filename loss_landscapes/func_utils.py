import pickle
import numpy as np
from typing import Any


def flatten(Ws1: np.array, bs1: np.array):
    lists_now = []

    for W_now in Ws1:
        lists_now.append(W_now.reshape([-1]))

    for b_now in bs1:
        lists_now.append(b_now.reshape([-1]))
    return np.concatenate(lists_now, axis=0)


def reform(flat1, Ws, bs):
    sofar = 0
    Ws_out_now = []
    bs_out_now = []
    for W in Ws:
        shape_now = W.get_shape().as_list()
        size_now = np.prod(shape_now)
        elements = flat1[sofar : sofar + size_now]
        sofar = sofar + size_now
        Ws_out_now.append(np.array(elements).reshape(shape_now))
    for b in bs:
        shape_now = b.get_shape().as_list()
        size_now = np.prod(shape_now)
        elements = flat1[sofar : sofar + size_now]
        sofar = sofar + size_now
        bs_out_now.append(np.array(elements).reshape(shape_now))
    return Ws_out_now, bs_out_now


def save_data(data: Any, name: str):
    with open(name, "wb") as f:
        pickle.dump(data, f)


def load_data(file_name: str):

    try:
        with open(file_name, "rb") as f:
            x = pickle.load(f)
    except:
        Exception(f"File not Found {file_name}")
    return x
