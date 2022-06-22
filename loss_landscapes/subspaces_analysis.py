import argparse
from optparse import Option
import torch
import numpy as np
from utils.printing import print_header
from typing import Tuple, Optional
from loss_landscapes.func_utils import (
    flatten,
    load_data,
    reform,
    get_all_models_metrics,
    average_var,
    save_data,
)
from loss_landscapes.func_sampling import (
    get_gaussian_sample,
    get_pca_gaussian_flat_sampling,
    get_rand_norm_direction,
)
from sklearn.decomposition import PCA
from loss_landscapes.models.MediumCNN import MediumCNN
from loss_landscapes.plot_subspace_analysis import (
    plot_ensemble_accuracy_test,
    plot_ensemble_brier_test,
)
from loss_landscapes.paths import (
    BS_BY_EPOCHS,
    BS_DIAGONAL,
    BS_MANY,
    BS_PCA_GAUSSIAN,
    BS_PCA_RAND_GAUSSIAN,
    BS_RAND,
    DIAG_GAUS_PRED,
    ORIG_PRED,
    PCA_GAUS_PRED,
    PCA_RAND_PRED,
    RAND_PRED,
    TEST_SET,
    VALIDATION_SET,
    WA_PRED,
    WS_BY_EPOCHS,
    WS_DIAGONAL,
    WS_MANY,
    WS_PCA_GAUSSIAN,
    WS_PCA_RAND_GAUSSIAN,
    WS_RAND,
)
from utils.pytorch_custom import get_labels_from_dataloader


def get_low_rank_approximation_of_the_random_samplings(
    points_to_collect: int,
    rand_Ws: np.ndarray,
    rand_bs: np.ndarray,
    load_saved: Optional[bool] = False,
) -> list:
    """PCA low-rank approximation of the random samplings"""
    if load_saved:
        pca_gaussian_rand_Ws = load_data(WS_PCA_RAND_GAUSSIAN)
        pca_gaussian_rand_bs = load_data(BS_PCA_RAND_GAUSSIAN)
    else:
        pca_gaussian_rand_Ws = [[] for _ in range(points_to_collect)]
        pca_gaussian_rand_bs = [[] for _ in range(points_to_collect)]

        for mid in range(points_to_collect):
            Ws_traj = rand_Ws[mid]
            bs_traj = rand_bs[mid]

            vs_list = []
            for i in range(len(Ws_traj)):
                vs_list.append(flatten(Ws_traj[i], bs_traj[i]))

            vs_np = np.stack(vs_list, axis=0)

            means = np.mean(vs_np, axis=0)
            stds = np.std(vs_np, axis=0)
            vs_np_centered = vs_np - means.reshape([1, -1])

            pca = PCA(n_components=rank)

            pca.fit(vs_np_centered)
            for i in range(num_sample):
                v_sample = get_pca_gaussian_flat_sampling(pca, means, rank, scale=1.0)
                w_sample, b_sample = reform(v_sample, Ws_traj[0], bs_traj[0])

                pca_gaussian_rand_Ws[mid].append(w_sample)
                pca_gaussian_rand_bs[mid].append(b_sample)
        save_data(pca_gaussian_rand_bs, BS_PCA_RAND_GAUSSIAN)
        save_data(pca_gaussian_rand_Ws, WS_PCA_RAND_GAUSSIAN)

    return pca_gaussian_rand_Ws, pca_gaussian_rand_bs


def get_torch_state_dict_from_list_of_weights_and_biases(
    weights: np.ndarray, biases: np.ndarray, architecture: list
) -> dict:
    """given a list of weights and biases creates the sate dict"""
    state_dict = {}
    for weight, bias, layer in zip(weights, biases, architecture):
        state_dict[f"{layer}.weight"] = torch.Tensor(weight)
        state_dict[f"{layer}.bias"] = torch.Tensor(bias)

    return state_dict


def get_pred_list(
    Ws_list: list, bs_list: list, testset: torch.utils.data.dataloader.DataLoader
):
    """Given a list of model weights, feed_dict and a session,
    returns the model predictions as a list."""
    pred_list = []
    for id in range(len(Ws_list)):
        Ws_now = Ws_list[id]
        bs_now = bs_list[id]
        state_dict = get_torch_state_dict_from_list_of_weights_and_biases(
            Ws_now, bs_now, architecture=MediumCNN.architecture
        )
        cnn = MediumCNN()
        cnn.load_state_dict(state_dict)
        pred_eval_out, acc, _ = cnn.get_validation_predictions(testset)
        print(f"Run Id {id} Test Accuracy: {acc:.3f}%")
        pred_list.append(pred_eval_out)
    return pred_list


def get_subspace_pred_list(
    Ws_subspace_list: list,
    bs_subspace_list: list,
    testloader: torch.utils.data.dataloader.DataLoader,
):
    """Consider a list of subspaces, each has a list of sampled weights.
    This function computes model predictions, ensembles the predictions
    within each subspace, and returns the list of ensembled predictions."""
    subspace_pred = []
    num_subspace = len(Ws_subspace_list)

    for mid in range(num_subspace):
        pred_list_now = get_pred_list(
            Ws_subspace_list[mid], bs_subspace_list[mid], testloader
        )
        subspace_pred.append(np.mean(np.stack(pred_list_now, axis=0), axis=0))
    return subspace_pred


def get_random_subspace_sampling(
    points_to_collect: int, num_sample: int, load_saved: Optional[bool] = False
) -> Tuple[list, list]:
    """perturbates the loss landscape with a random direction

    Returns:
        The perturbed weights and biases
    """

    if load_saved:
        rand_Ws = load_data(WS_RAND)
        rand_bs = load_data(BS_RAND)
    else:
        # Random samples need to meet this accuracy threshold to be included.
        acc_threshold = 0.70

        rand_Ws = [[] for _ in range(points_to_collect)]
        rand_bs = [[] for _ in range(points_to_collect)]
        Ws_many = load_data(WS_MANY)
        bs_many = load_data(BS_MANY)
        validator = torch.load(VALIDATION_SET)
        for mid in range(points_to_collect):
            for _ in range(num_sample):

                # create the weight and bias space
                vs = flatten(Ws_many[mid], bs_many[mid])
                for trial in range(5):
                    scale = 10 * np.random.uniform()

                    # move weight and bias space to random a direction
                    # \theta + tv
                    v_sample = vs + scale * get_rand_norm_direction(vs.shape)

                    # rebuild the weights and bias sample
                    w_sample, b_sample = reform(v_sample, Ws_many[mid], bs_many[mid])
                    state_dict = get_torch_state_dict_from_list_of_weights_and_biases(
                        w_sample,
                        b_sample,
                        architecture=MediumCNN.architecture,
                    )
                    cnn = MediumCNN()
                    cnn.load_state_dict(state_dict)

                    _, val_acc, _ = cnn.get_validation_predictions(validator)
                    if val_acc >= acc_threshold:
                        rand_Ws[mid].append(w_sample)
                        rand_bs[mid].append(b_sample)
                        print(
                            "Obtaining 1 rand sample at scale {} with validation acc {} at {}th try".format(
                                scale, val_acc, trial
                            )
                        )
                        break
                    if trial == 4:
                        print("No luck -------------------")
        save_data(rand_Ws, WS_RAND)
        save_data(rand_bs, BS_RAND)
    return rand_Ws, rand_bs


def get_diagonal_and_low_rank_gaussians_subspaces(
    rank: int,
    num_sample: int,
    points_to_collect: int,
    load_saved: Optional[bool] = False,
) -> Tuple[list, list, list, list]:

    if load_saved:
        dial_gaussian_whole_bs = load_data(BS_DIAGONAL)
        dial_gaussian_whole_Ws = load_data(WS_DIAGONAL)

        pca_gaussian_whole_bs = load_data(BS_PCA_GAUSSIAN)
        pca_gaussian_whole_Ws = load_data(WS_PCA_GAUSSIAN)
    else:
        # last 10 epochs collected during training
        Ws_by_epochs_many = load_data(WS_BY_EPOCHS)
        bs_by_epochs_many = load_data(BS_BY_EPOCHS)

        dial_gaussian_whole_Ws = [[] for _ in range(points_to_collect)]
        dial_gaussian_whole_bs = [[] for _ in range(points_to_collect)]

        pca_gaussian_whole_Ws = [[] for _ in range(points_to_collect)]
        pca_gaussian_whole_bs = [[] for _ in range(points_to_collect)]

        # for each independent run
        for mid in range(points_to_collect):
            Ws_traj = Ws_by_epochs_many[mid]
            bs_traj = bs_by_epochs_many[mid]

            vs_list = []
            for i in range(len(Ws_traj)):
                vs_list.append(flatten(Ws_traj[i], bs_traj[i]))

            vs_np = np.stack(vs_list, axis=0)

            # bias and weights means and stds over 10 runs
            means = np.mean(vs_np, axis=0)
            stds = np.std(vs_np, axis=0)
            vs_np_centered = vs_np - means.reshape([1, -1])

            pca = PCA(n_components=rank)
            pca.fit(vs_np_centered)

            for i in range(num_sample):
                v_sample = get_gaussian_sample(means, stds, scale=1.0)
                w_sample, b_sample = reform(v_sample, Ws_traj[0], bs_traj[0])

                # Diagonal Gaussian Subspace
                dial_gaussian_whole_Ws[mid].append(w_sample)
                dial_gaussian_whole_bs[mid].append(b_sample)

                v_sample = get_pca_gaussian_flat_sampling(pca, means, rank, scale=1.0)
                w_sample, b_sample = reform(v_sample, Ws_traj[0], bs_traj[0])

                # Low-Rank Gaussian Subspace
                pca_gaussian_whole_Ws[mid].append(w_sample)
                pca_gaussian_whole_bs[mid].append(b_sample)
        save_data(dial_gaussian_whole_bs, BS_DIAGONAL)
        save_data(dial_gaussian_whole_Ws, WS_DIAGONAL)

        save_data(pca_gaussian_whole_bs, BS_PCA_GAUSSIAN)
        save_data(pca_gaussian_whole_Ws, WS_PCA_GAUSSIAN)
    return (
        dial_gaussian_whole_Ws,
        dial_gaussian_whole_bs,
        pca_gaussian_whole_Ws,
        pca_gaussian_whole_bs,
    )


def get_original_predictions(
    Ws_many: list,
    bs_many: list,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: bool,
) -> list:
    """Get original predictions on test data."""
    print_header("Original Preds")

    if load_saved:
        orig_pred = load_data(ORIG_PRED)
    else:
        orig_pred = get_pred_list(Ws_many, bs_many, testloader)
        save_data(orig_pred, ORIG_PRED)
    return orig_pred


def get_weights_average_predictions(
    points_to_collect: int,
    Ws_by_epochs_many: list,
    bs_by_epochs_many,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: bool,
) -> list:

    """Compute averaged weights predictions"""
    print_header("Average Weights Preds")
    if load_saved:
        wa_pred = load_data(WA_PRED)
    else:
        wa_Ws = [[] for _ in range(points_to_collect)]
        wa_bs = [[] for _ in range(points_to_collect)]
        for i in range(points_to_collect):

            wa_Ws[i] = average_var(Ws_by_epochs_many[i])
            wa_bs[i] = average_var(bs_by_epochs_many[i])
        wa_pred = get_pred_list(wa_Ws, wa_bs, testloader)
        save_data(wa_pred, WA_PRED)
    return wa_pred


def get_diagonal_gaussian_subspace_predictions(
    dial_gaussian_whole_Ws: list,
    dial_gaussian_whole_bs: list,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: Optional[bool] = False,
):
    """get diagonal gaussian subspace prediction"""
    print_header("Diag Gaussian Subspace Preds")
    if load_saved:
        diag_gaus_pred = load_data(DIAG_GAUS_PRED)
    else:
        diag_gaus_pred = get_subspace_pred_list(
            dial_gaussian_whole_Ws, dial_gaussian_whole_bs, testloader
        )
        save_data(diag_gaus_pred, DIAG_GAUS_PRED)
    return diag_gaus_pred


def get_pca_gaus_pred(
    pca_gaussian_whole_Ws: list,
    pca_gaussian_whole_bs: list,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: Optional[bool] = False,
):
    # low rank gaussian
    print_header("Low-Rank Gaussian Subspace Preds")
    if load_saved:
        pca_gaus_pred = load_data(PCA_GAUS_PRED)
    else:
        pca_gaus_pred = get_subspace_pred_list(
            pca_gaussian_whole_Ws, pca_gaussian_whole_bs, testloader
        )
        save_data(pca_gaus_pred, PCA_GAUS_PRED)
    return pca_gaus_pred


def get_pca_gaus_rand_pred(
    pca_gaussian_rand_Ws: list,
    pca_gaussian_rand_bs: list,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: Optional[bool] = False,
) -> list:
    # dropout?
    print_header("Dropout Subspace Preds")

    if load_saved:
        pca_rand_pred = load_data(PCA_RAND_PRED)
    else:
        pca_rand_pred = get_subspace_pred_list(
            pca_gaussian_rand_Ws, pca_gaussian_rand_bs, testloader
        )
        save_data(pca_rand_pred, PCA_RAND_PRED)
    return pca_rand_pred


def get_rand_pred(
    rand_Ws: list,
    rand_bs: list,
    testloader: torch.utils.data.dataloader.DataLoader,
    load_saved: Optional[bool] = False,
) -> list:
    # random subspace sampling
    print_header("Random Subspace Sampling Preds")

    if load_saved:
        rand_pred = load_data(RAND_PRED)
    else:
        rand_pred = get_subspace_pred_list(rand_Ws, rand_bs, testloader)
        save_data(rand_pred, RAND_PRED)
    return rand_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load-saved-pred",
        dest="load_saved_pred",
        action="store_true",
        help="whether we should load predictions",
    )
    parser.set_defaults(load_saved_pred=False)
    parser.add_argument(
        "--load-saved-wb",
        dest="load_saved_wb",
        action="store_true",
        help="whether we should load whether bias",
    )
    parser.set_defaults(load_saved_wb=False)
    args = parser.parse_args()

    # PCA rank.
    rank = 5
    num_sample = 30
    points_to_collect = 5
    testloader = torch.load(TEST_SET)
    y_test = get_labels_from_dataloader(testloader).numpy()
    # ************************
    # Computationally intensive
    # Better save the data
    # *************************
    (
        dial_gaussian_whole_Ws,
        dial_gaussian_whole_bs,
        pca_gaussian_whole_Ws,
        pca_gaussian_whole_bs,
    ) = get_diagonal_and_low_rank_gaussians_subspaces(
        rank, num_sample, points_to_collect, args.load_saved_wb
    )

    rand_Ws, rand_bs = get_random_subspace_sampling(
        points_to_collect, num_sample, args.load_saved_wb
    )
    (
        pca_gaussian_rand_Ws,
        pca_gaussian_rand_bs,
    ) = get_low_rank_approximation_of_the_random_samplings(
        points_to_collect, rand_Ws, rand_bs, args.load_saved_wb
    )

    pca_rand_pred = get_pca_gaus_rand_pred(
        pca_gaussian_rand_Ws, pca_gaussian_rand_bs, testloader, args.load_saved_pred
    )

    Ws_by_epochs_many = load_data(WS_BY_EPOCHS)
    bs_by_epochs_many = load_data(BS_BY_EPOCHS)
    Ws_many = load_data(WS_MANY)
    bs_many = load_data(BS_MANY)

    rand_pred = get_rand_pred(rand_Ws, rand_bs, testloader, args.load_saved_pred)

    orig_pred = get_original_predictions(
        Ws_many, bs_many, testloader, args.load_saved_pred
    )
    wa_pred = get_weights_average_predictions(
        points_to_collect,
        Ws_by_epochs_many,
        bs_by_epochs_many,
        testloader,
        args.load_saved_pred,
    )
    diag_gaus_pred = get_diagonal_gaussian_subspace_predictions(
        dial_gaussian_whole_Ws, dial_gaussian_whole_bs, testloader, args.load_saved_pred
    )

    pca_gaus_pred = get_pca_gaus_pred(
        pca_gaussian_whole_Ws, pca_gaussian_whole_bs, testloader, args.load_saved_pred
    )

    max_ens_size = points_to_collect - 1

    orig_metrics = get_all_models_metrics(orig_pred, y_test, max_ens_size=max_ens_size)
    print("Got Origin Metrics")
    wa_metrics = get_all_models_metrics(wa_pred, y_test, max_ens_size=max_ens_size)
    print("Got Weighted Average Metrics")
    diag_metrics = get_all_models_metrics(
        diag_gaus_pred, y_test, max_ens_size=max_ens_size
    )
    print("Got Diagonal Gaussian Metrics")
    pca_metrics = get_all_models_metrics(
        pca_gaus_pred, y_test, max_ens_size=max_ens_size
    )
    print("Got Random Metrics")
    rand_metrics = get_all_models_metrics(rand_pred, y_test, max_ens_size=max_ens_size)
    print("Got PCA Rand Metrics")
    pca_rand_metrics = get_all_models_metrics(
        pca_rand_pred, y_test, max_ens_size=max_ens_size
    )

    plot_ensemble_accuracy_test(
        max_ens_size,
        orig_metrics,
        diag_metrics,
        pca_metrics,
        wa_metrics,
        rand_metrics,
        pca_rand_metrics,
    )

    plot_ensemble_brier_test(
        max_ens_size,
        orig_metrics,
        diag_metrics,
        pca_metrics,
        wa_metrics,
        rand_metrics,
        pca_rand_metrics,
    )
