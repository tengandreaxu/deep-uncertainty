import numpy as np
import matplotlib.pyplot as plt


def plot_ensemble_brier_test(
    max_ens_size: int,
    orig_metrics: dict,
    diag_metrics: dict,
    pca_metrics: dict,
    wa_metrics: dict,
    rand_metrics: dict,
    pca_rand_metrics: dict,
):

    title = "Ensemble Brier test"

    plt.xlabel("Ensemble size")
    plt.ylabel("Test Brier")

    ensemble_sizes = np.asarray(range(max_ens_size)) + 1
    plt.plot(
        ensemble_sizes,
        orig_metrics["brier"]["ensemble"],
        marker="s",
        label="probs ensembling",
        color="navy",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(orig_metrics["brier"]["individual"])] * len(ensemble_sizes),
        label="original",
        color="blue",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(diag_metrics["brier"]["individual"])] * len(ensemble_sizes),
        label="Diag",
        color="pink",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(pca_metrics["brier"]["individual"])] * len(ensemble_sizes),
        label="PCA",
        color="green",
    )

    plt.plot(
        ensemble_sizes,
        diag_metrics["brier"]["ensemble"],
        marker="s",
        label="Diag Ensemble",
        color="red",
    )

    plt.plot(
        ensemble_sizes,
        pca_metrics["brier"]["ensemble"],
        marker="s",
        label="PCA Ensemble",
        color="green",
    )

    plt.plot(
        ensemble_sizes,
        wa_metrics["brier"]["ensemble"],
        marker="s",
        label="WA ensembling",
        color="grey",
    )

    plt.plot(
        ensemble_sizes,
        rand_metrics["brier"]["ensemble"],
        marker="s",
        label="Rand Ensemble",
        color="yellow",
    )
    plt.plot(
        ensemble_sizes,
        pca_rand_metrics["brier"]["ensemble"],
        marker="s",
        label="PCA Rand Ensemble",
        color="m",
    )

    plt.xlim(1, max_ens_size)
    plt.legend()
    plt.savefig("plots/loss_landscapes/brier_landscapes.png")
    plt.close()


def plot_ensemble_accuracy_test(
    max_ens_size: int,
    orig_metrics: dict,
    diag_metrics: dict,
    pca_metrics: dict,
    wa_metrics: dict,
    rand_metrics: dict,
    pca_rand_metrics: dict,
):
    title = "Ensemble ACC test"

    plt.xlabel("Ensemble size")
    plt.ylabel("Test Acc")

    ensemble_sizes = np.asarray(range(max_ens_size)) + 1
    plt.plot(
        ensemble_sizes,
        orig_metrics["accuracy"]["ensemble"],
        marker="s",
        label="probs ensembling",
        color="navy",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(orig_metrics["accuracy"]["individual"])] * len(ensemble_sizes),
        label="original",
        color="blue",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(diag_metrics["accuracy"]["individual"])] * len(ensemble_sizes),
        label="Diag",
        color="pink",
    )

    plt.plot(
        ensemble_sizes,
        [np.mean(pca_metrics["accuracy"]["individual"])] * len(ensemble_sizes),
        label="PCA",
        color="green",
    )

    plt.plot(
        ensemble_sizes,
        diag_metrics["accuracy"]["ensemble"],
        marker="s",
        label="Diag+Ensemble",
        color="red",
    )
    plt.plot(
        ensemble_sizes,
        wa_metrics["accuracy"]["ensemble"],
        marker="s",
        label="WA+ensembling",
        color="grey",
    )

    plt.plot(
        ensemble_sizes,
        pca_metrics["accuracy"]["ensemble"],
        marker="s",
        label="PCA+Ensemble",
        color="green",
    )
    plt.plot(
        ensemble_sizes,
        rand_metrics["accuracy"]["ensemble"],
        marker="s",
        label="Rand+Ensemble",
        color="yellow",
    )
    plt.plot(
        ensemble_sizes,
        pca_rand_metrics["accuracy"]["ensemble"],
        marker="s",
        label="PCA Rand+Ensemble",
        color="m",
    )

    plt.legend()
    plt.xlim(1, max_ens_size)
    plt.savefig("plots/loss_landscapes/accuracies_landscapes.png")
    plt.close()
