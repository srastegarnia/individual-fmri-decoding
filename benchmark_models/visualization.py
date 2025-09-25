import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import plotting  # needed for plotting.view_img

# Exported API
__all__ = [
    "classifier_history",
    "conf_matrix",
    "plot_cv_indices",
    "linear_decoder_weights",
]

# Sensible default colormaps for plot_cv_indices
_CMAP_CV = plt.cm.coolwarm
_CMAP_DATA = plt.cm.Pastel1


def _ensure_dir(path_str: str) -> None:
    """Create directory for a file path if it doesn't exist."""
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def classifier_history(history, title, results_outpath, output_file_name):
    """
    Plot training history curves (accuracy, loss) for Keras-style History objects.

    Parameters
    ----------
    history : object with .history dict (e.g., keras.callbacks.History)
    title : str
    results_outpath : str
    output_file_name : str
    """
    # Accuracy
    acc_path = os.path.join(results_outpath, f"{output_file_name}_model_accuracy.png")
    _ensure_dir(acc_path)

    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{title} model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Loss
    loss_path = os.path.join(results_outpath, f"{output_file_name}_model_loss.png")
    _ensure_dir(loss_path)

    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{title} model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()


def conf_matrix(
    model_cm,
    unique_conditions,
    title,
    cm_results_outpath,
    output_file_name,
    decoder,
    subject,
    region_approach,
    resolution,
    HRFlag_process,
):
    """
    Save confusion matrix CSV, append diagonal to a summary CSV, and render a heatmap.

    Parameters
    ----------
    model_cm : array-like (n_classes, n_classes) — values already normalized or raw
    unique_conditions : list[str] — label order used in model_cm
    title : str
    cm_results_outpath : str — directory to write outputs
    output_file_name : str — base name (no extension)
    decoder, subject, region_approach, resolution, HRFlag_process : metadata
    """
    df_cm = pd.DataFrame(model_cm, index=unique_conditions, columns=unique_conditions)

    # Append per-class accuracies (diag) to results_summary.csv
    cm_diag = np.diag(df_cm.values, k=0)
    summary_row = np.append(
        [subject, decoder, region_approach, resolution, HRFlag_process], cm_diag
    )

    summary_csv = os.path.join(cm_results_outpath, "results_summary.csv")
    _ensure_dir(summary_csv)
    with open(summary_csv, "a+", newline="") as write_obj:
        from csv import writer as csv_writer

        csv_writer(write_obj).writerow(summary_row)

    # Save matrix as CSV
    cm_csv = os.path.join(cm_results_outpath, f"{output_file_name}.csv")
    df_cm.to_csv(cm_csv)

    # Plot heatmap
    plt.figure(figsize=(20, 14))
    ax = sns.heatmap(df_cm, annot=True, cmap="Blues", square=True, fmt=".2f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel("true labels", fontsize=14, fontweight="bold")
    plt.ylabel("predicted labels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """
    Visualize cross-validation splits (training/test indices, classes, groups).

    Parameters
    ----------
    cv : a scikit-learn CV splitter with .split(X, y, groups)
    X, y, group : arrays
    ax : matplotlib axes
    n_splits : int
    lw : int — line width
    """
    import numpy as np  # local to avoid polluting module namespace

    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=_CMAP_CV,
            vmin=-0.2,
            vmax=1.2,
        )

    # Class and group bars
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=_CMAP_DATA)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=_CMAP_DATA)

    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, max(100, len(X))],
    )
    ax.set_title(f"{type(cv).__name__}", fontsize=15)
    return ax


# ----------------------------------------------------------------------------------------------------
def linear_decoder_weights(model_svm, masker=None):
    """
    Visualize linear model weights in image space.

    Parameters
    ----------
    model_svm : fitted linear SVM-like estimator with .coef_
    masker : NiftiMasker-like (optional)
        If None, tries to use a global `masker` defined by the caller's scope.

    Notes
    -----
    Kept compatible with older usage that relied on a global `masker`.
    """
    if masker is None:
        raise ValueError("masker is required to inverse_transform model weights.")
    coef_img = masker.inverse_transform(model_svm.coef_[0, :])
    display = plotting.view_img(
        coef_img, title="SVM weights map", dim=-1, resampling_interpolation="nearest"
    )
    return display