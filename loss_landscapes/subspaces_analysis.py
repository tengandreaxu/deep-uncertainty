import numpy as np
from loss_landscapes.func_utils import flatten, load_data
from loss_landscapes.func_sampling import (
    get_gaussian_sample,
    get_pca_gaussian_flat_sampling,
)
from sklearn.decomposition import PCA

from loss_landscapes.paths import BS_BY_EPOCHS, WS_BY_EPOCHS


if __name__ == "__main__":
    # PCA rank.
    rank = 5
    num_sample = 30
    points_to_collect = 5

    Ws_by_epochs_many = load_data(WS_BY_EPOCHS)
    bs_by_epochs_many = load_data(BS_BY_EPOCHS)
    dial_gaussian_whole_Ws = [[] for _ in range(points_to_collect)]
    dial_gaussian_whole_bs = [[] for _ in range(points_to_collect)]

    pca_gaussian_whole_Ws = [[] for _ in range(points_to_collect)]
    pca_gaussian_whole_bs = [[] for _ in range(points_to_collect)]

    for mid in range(points_to_collect):
        Ws_traj = Ws_by_epochs_many[mid]
        bs_traj = bs_by_epochs_many[mid]

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
            v_sample = get_gaussian_sample(means, stds, scale=1.0)
            w_sample, b_sample = reform(v_sample)
            dial_gaussian_whole_Ws[mid].append(w_sample)
            dial_gaussian_whole_bs[mid].append(b_sample)

            v_sample = get_pca_gaussian_flat_sampling(pca, means, rank, scale=1.0)
            w_sample, b_sample = reform(v_sample)

            pca_gaussian_whole_Ws[mid].append(w_sample)
            pca_gaussian_whole_bs[mid].append(b_sample)
