"""Writes JSON/CSV artifacts and plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    """Write deterministic, human-readable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def plot_residuals(path, y_true, y_pred, code):
    res = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, res, alpha=0.7)
    ax.axhline(0, color="red", lw=1)
    ax.set(xlabel="Predicted yield (tons)", ylabel="Residual (actual - predicted)",
           title=f"Residuals on test set  [{code}]")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(path, y_true, y_pred, code):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1],
                                            display_labels=["not high risk", "high risk"], ax=ax)
    ax.set_title(f"Confusion matrix  [{code}]")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_clusters(path, Z, labels, code):
    xy = PCA(n_components=2, random_state=0).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="viridis", alpha=0.8)
    ax.legend(*sc.legend_elements(), title="cluster")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Farm clusters (PCA view, train)  [{code}]")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
