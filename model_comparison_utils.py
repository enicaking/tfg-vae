# -*- coding: utf-8 -*-
"""
model_comparison_utils.py

Refactor of the analysis cells that originally lived under the markdown header:
    "# Compare latent spaces, reconstruction quality, and verify correct
     implementation of uncoded dSCVI"

Everything below is turned into reusable functions so you can:
  - compare N models (not just cont_model vs uncoded_model) on the same plot
  - register a model's name ONCE and have every plot/model file for it use
    that name consistently, in a fixed folder, without ever overwriting a
    previous model's outputs.

USAGE PATTERN (drop this in a cell after training a new model)
----------------------------------------------------------------
    register_model("uncoded_dscvi_v2")          # (1) name it once
    save_model(uncoded_model, "uncoded_dscvi_v2")  # (2) save checkpoint

    models = {
        "scVI (continuous)": cont_model,
        "uncoded_dscvi_v1": uncoded_model_v1,
        "uncoded_dscvi_v2": uncoded_model_v2,
    }
    plot_loss_metrics(models)
    latents = {name: get_latent(m, adata, binarize=True) for name, m in models.items()}
    plot_umap_comparison(latents)
    plot_latent_heatmap(uncoded_model_v2, adata, model_name="uncoded_dscvi_v2")
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from sklearn.cluster import KMeans


# =====================================================================
# 1. DIRECTORY / NAMING CONSTANTS
# =====================================================================
# These assume `save_dir` and `notebook_dir` were already defined earlier
# in the notebook (they are, in the IMPORT section). We build everything
# underneath those so model checkpoints and plots live in predictable,
# separate subfolders on your Drive.

# Fall back to sensible defaults if this module is imported before those
# variables exist in the notebook namespace.
_BASE_DIR = globals().get("notebook_dir", "/content/drive/MyDrive/TFG")
_SAVE_DIR = globals().get("save_dir", os.path.join(_BASE_DIR, "scvi_data"))

MODELS_DIR = os.path.join(_SAVE_DIR, "models")
PLOTS_DIR = os.path.join(_BASE_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Every model you train should get ONE name, set once, e.g. right after
# you instantiate it. Everything else (model checkpoint folder, plot file
# names) derives from this so nothing collides across experiments.
DEFAULT_MODEL_NAME = "uncoded_dscvi_v1"

# Keeps track of every name you've registered this session, purely so
# `list_registered_models()` can remind you what you've already used.
_REGISTERED_MODELS: set[str] = set()


def register_model(model_name: str) -> str:
    """
    Call this once per new model/experiment, right when you name it.

    Doesn't do anything magic -- it just validates the name, remembers it,
    and returns it back so you can do:
        MODEL_NAME = register_model("uncoded_dscvi_v2")
    and use MODEL_NAME everywhere below instead of retyping the string.
    """
    clean_name = model_name.strip().replace(" ", "_")
    if clean_name in _REGISTERED_MODELS:
        print(f"[register_model] Warning: '{clean_name}' was already "
              f"registered this session. Re-using it (existing plots for "
              f"this name will get an auto-incremented suffix, not "
              f"overwritten).")
    _REGISTERED_MODELS.add(clean_name)
    return clean_name


def list_registered_models() -> list[str]:
    return sorted(_REGISTERED_MODELS)


def model_path(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Consistent checkpoint folder for `model.save(...)` / `Model.load(...)`."""
    return os.path.join(MODELS_DIR, model_name)


def save_model(model, model_name: str = DEFAULT_MODEL_NAME, overwrite: bool = True):
    """Thin wrapper so saving always goes through the same naming rule."""
    path = model_path(model_name)
    model.save(path, overwrite=overwrite)
    print(f"Saved model '{model_name}' -> {path}")
    return path


def plot_path(plot_type: str,
              model_name: str = DEFAULT_MODEL_NAME,
              ext: str = "png",
              timestamp: bool = False) -> str:
    """
    Build a consistent, collision-free path for a plot file.

    e.g. plot_path("umap")                 -> plots/uncoded_dscvi_v1_umap.png
         plot_path("tsne", "my_model_v2")  -> plots/my_model_v2_tsne.png

    If a file with that exact name already exists, a numeric suffix is
    appended automatically (_1, _2, ...) so you never silently overwrite
    an earlier model's plot -- even if you forget to change model_name.
    """
    fname = f"{model_name}_{plot_type}"
    if timestamp:
        fname += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    candidate = os.path.join(PLOTS_DIR, f"{fname}.{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(PLOTS_DIR, f"{fname}_{counter}.{ext}")
        counter += 1
    return candidate


# =====================================================================
# 2. LATENT EXTRACTION
# =====================================================================
def get_latent(model,
               adata=None,
               give_mean: bool = True,
               binarize: bool = False,
               threshold: float = 0.5) -> np.ndarray:
    """
    Pull a latent representation out of any scVI-style model in a
    consistent way, optionally binarizing it into hard bits (as done for
    the discrete uncoded model in the original notebook).
    """
    kwargs = {"give_mean": give_mean}
    if adata is not None:
        kwargs["adata"] = adata
    latent = model.get_latent_representation(return_dist=False, **kwargs)
    if binarize:
        latent = np.where(latent > threshold, 1, 0)
    return latent


def per_bit_variance_report(model,
                             adata=None,
                             model_name: str = DEFAULT_MODEL_NAME,
                             alive_threshold: float = 0.01) -> pd.DataFrame:
    """
    Reports per-bit mean/variance and flags "alive" vs "possibly dead" bits
    (variance > alive_threshold), same diagnostic as the original notebook,
    generalized to any model.
    """
    qi = get_latent(model, adata=adata, give_mean=True, binarize=False)
    per_bit_mean = qi.mean(axis=0)
    per_bit_var = qi.var(axis=0)
    alive = per_bit_var > alive_threshold

    df = pd.DataFrame({
        "bit": [f"bit_{i}" for i in range(qi.shape[1])],
        "mean": per_bit_mean,
        "variance": per_bit_var,
        "alive": alive,
    })
    n_alive = int(alive.sum())
    print(f"[{model_name}] {n_alive}/{qi.shape[1]} bits show meaningful "
          f"variation across cells (variance > {alive_threshold})")
    return df


# =====================================================================
# 3. LOSS METRICS
# =====================================================================
def _plot_metric(history, train_key, val_key, title, ylabel, ax, model_name=""):
    if train_key in history and not history[train_key].empty:
        ax.plot(history[train_key].index, history[train_key].iloc[:, 0],
                 label=f"{model_name} train")
    if val_key in history and not history[val_key].empty:
        ax.plot(history[val_key].index, history[val_key].iloc[:, 0],
                 label=f"{model_name} validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_loss_metrics(models: dict,
                       metrics: list[tuple[str, str, str, str]] | None = None,
                       model_name: str = DEFAULT_MODEL_NAME,
                       save: bool = True,
                       show: bool = True):
    """
    Grid of loss curves: one overlay row with every model on the same
    axes (for direct comparison), followed by one row per model showing
    just that model's own curves (so a noisy/compressed curve isn't
    flattened by another model's scale).

    models: dict of {label: trained_model} (must expose `.history`)
    metrics: optional override list of
             (train_key, val_key, title, ylabel) tuples.
    model_name: used only for the saved filename (this plot compares
                several models, so it's usually a comparison-level name,
                e.g. "cont_vs_uncoded_v1_v2").
    """
    if metrics is None:
        metrics = [
            ("elbo_train", "elbo_validation", "ELBO Loss", "ELBO"),
            ("reconstruction_loss_train", "reconstruction_loss_validation",
             "Reconstruction Loss", "Reconstruction Loss"),
            ("kl_local_train", "kl_local_validation",
             "KL Local Divergence", "KL Divergence"),
        ]

    n_cols = len(metrics)
    n_rows = 1 + len(models)  # overlay row + one row per model

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)

    # --- Row 0: overlay of every model on the same axes ---
    for label, model in models.items():
        history = model.history
        for ax, (train_key, val_key, title, ylabel) in zip(axes[0], metrics):
            _plot_metric(history, train_key, val_key, title, ylabel, ax,
                         model_name=label)
    for ax in axes[0]:
        ax.legend()

    # --- One row per model, showing only that model's curves ---
    for row_idx, (label, model) in enumerate(models.items(), start=1):
        history = model.history
        for ax, (train_key, val_key, title, ylabel) in zip(axes[row_idx], metrics):
            _plot_metric(history, train_key, val_key, f"{title} — {label}",
                         ylabel, ax, model_name=label)
            ax.legend()

    plt.tight_layout()
    if save:
        path = plot_path("loss_metrics", model_name=model_name)
        plt.savefig(path)
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# =====================================================================
# 4. UMAP COMPARISON
# =====================================================================
def plot_umap_comparison(latents: dict,
                          model_name: str = DEFAULT_MODEL_NAME,
                          color_by: np.ndarray | pd.Series | None = None,
                          color_label: str = "",
                          save: bool = True,
                          show: bool = True):
    """
    Compute and plot UMAP side-by-side for any number of latent spaces.

    latents: dict of {label: latent_array (n_cells x n_latent)}
    color_by: optional array/Series (e.g. cluster labels or cell types) to
              color all panels by, same length as n_cells.
    """
    if umap is None:
        raise ImportError("umap-learn is not installed. `pip install umap-learn`.")

    n = len(latents)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (label, latent) in zip(axes, latents.items()):
        embedding = umap.UMAP().fit_transform(latent)
        if color_by is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2,
                                  alpha=0.6, c=pd.Categorical(color_by).codes,
                                  cmap="tab20")
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    if color_by is not None:
        cbar = plt.colorbar(scatter, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label(color_label or "color")

    plt.tight_layout()
    if save:
        path = plot_path("umap_comparison", model_name=model_name)
        plt.savefig(path)
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_umap_kmeans(latent: np.ndarray,
                      model_name: str = DEFAULT_MODEL_NAME,
                      n_clusters: int = 10,
                      adata=None,
                      obs_key: str = "kmeans_clusters",
                      save: bool = True,
                      show: bool = True):
    """
    UMAP of a single latent space colored by k-means clusters computed on
    that same latent space (as done for the discrete bit latent originally).
    """
    if umap is None:
        raise ImportError("umap-learn is not installed. `pip install umap-learn`.")

    embedding = umap.UMAP().fit_transform(latent)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(latent)

    if adata is not None:
        adata.obs[obs_key] = cluster_labels.astype(str)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.5,
                          c=cluster_labels, cmap="viridis")
    ax.set_title(f"UMAP of {model_name} latent with K-means clusters")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("K-means cluster")

    plt.tight_layout()
    if save:
        path = plot_path("umap_kmeans", model_name=model_name)
        plt.savefig(path)
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, cluster_labels


# =====================================================================
# 5. T-SNE (scanpy-based + manual ellipse variant)
# =====================================================================
def compute_tsne(adata, use_rep: str, perplexity: int = 10):
    """Thin wrapper so every model's t-SNE is computed the same way."""
    sc.tl.tsne(adata, use_rep=use_rep, perplexity=perplexity, n_jobs=-1)
    return adata.obsm["X_tsne"]


def plot_tsne_scanpy(adata,
                      color: list[str],
                      model_name: str = DEFAULT_MODEL_NAME,
                      save: bool = True,
                      show: bool = True):
    """Standard scanpy t-SNE panel (cell_type / donor / cell_source, etc.)."""
    sc.pl.tsne(adata, color=color, ncols=len(color), frameon=False, show=False)
    if save:
        path = plot_path("tsne_scanpy", model_name=model_name)
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close()


def plot_tsne_with_ellipses(adata,
                             use_rep: str,
                             groupby: str = "cell_type",
                             model_name: str = DEFAULT_MODEL_NAME,
                             xlim: tuple[float, float] | None = None,
                             ylim: tuple[float, float] | None = None,
                             save: bool = True,
                             show: bool = True):
    """
    t-SNE scatter with per-category mean marker + 1/2-sigma covariance
    ellipses, generalized from the original notebook cell so it works for
    any latent representation stored under `use_rep` in adata.obsm.
    """
    tsne_coords = compute_tsne(adata, use_rep=use_rep)
    categories = adata.obs[groupby].cat.categories

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))

    # Background points
    for cat in categories:
        mask = adata.obs[groupby] == cat
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[color_map[cat]], s=5, alpha=0.3)

    # Foreground: mean + ellipses
    for cat in categories:
        mask = adata.obs[groupby] == cat
        points = tsne_coords[mask]

        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        ax.scatter(mean[0], mean[1], c="black", s=20, zorder=6)
        ax.annotate(cat, mean, fontsize=8, xytext=(4, 4),
                    textcoords="offset points")

        for n_std, alpha in [(1, 0.9), (2, 0.5)]:
            width, height = 2 * n_std * np.sqrt(vals)
            ellipse = mpatches.Ellipse(
                xy=mean, width=width, height=height, angle=angle,
                fill=False, edgecolor=color_map[cat], linewidth=1.5,
                alpha=alpha, zorder=5,
            )
            ax.add_patch(ellipse)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE ({model_name}) with class mean and covariance ellipses")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    if save:
        path = plot_path("tsne_ellipses", model_name=model_name, ext="jpeg")
        plt.savefig(path)
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# =====================================================================
# 6. LATENT SPACE INTERPRETABILITY HEATMAPS
# =====================================================================
def plot_latent_heatmap(model,
                         adata,
                         groupby: str = "cell_type",
                         model_name: str = DEFAULT_MODEL_NAME,
                         save: bool = True,
                         show: bool = True):
    """
    Three-panel heatmap: raw average bit activation per cell type,
    deviation from the global mean, and z-scored deviation -- generalized
    from the original single-model cell so it can be called per model.

    Returns (per_type_mean, deviation, zscore) DataFrames in case you want
    to inspect specific bits/cell types programmatically afterward.
    """
    qi_full = model.get_latent_representation(adata=adata, give_mean=True)
    bit_cols = [f"bit_{i}" for i in range(qi_full.shape[1])]
    df = pd.DataFrame(qi_full, columns=bit_cols)
    df[groupby] = adata.obs[groupby].values

    global_mean = df[bit_cols].mean(axis=0)
    global_std = df[bit_cols].std(axis=0)
    per_type_mean = df.groupby(groupby)[bit_cols].mean()
    deviation = per_type_mean.subtract(global_mean, axis=1)
    zscore = deviation.divide(global_std, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.heatmap(per_type_mean, annot=True, fmt=".2f", cmap="viridis", ax=axes[0],
                cbar_kws={"label": "Mean qi"})
    axes[0].set_title(f"Raw average bit activation per {groupby} ({model_name})")

    sns.heatmap(deviation, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1],
                cbar_kws={"label": "Deviation from global mean"})
    axes[1].set_title("Deviation from global average")

    sns.heatmap(zscore, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=axes[2],
                cbar_kws={"label": "Z-score"})
    axes[2].set_title("Z-scored deviation (std units)")

    for i, ax in enumerate(axes):
        ax.set_xlabel("Bit")
        ax.set_ylabel(groupby if i == 0 else "")

    plt.tight_layout()
    if save:
        path = plot_path("latent_heatmap", model_name=model_name)
        plt.savefig(path)
        print(f"Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)

    return per_type_mean, deviation, zscore


def max_abs_zscore_per_bit(zscore: pd.DataFrame) -> pd.Series:
    """
    Cleaner "dead vs alive" metric than global variance/mean alone: the
    max |z-score| per bit across all cell types. A bit is only worth
    calling dead if this is small (e.g. < 0.3-0.4) for every cell type.
    """
    return zscore.abs().max(axis=0).sort_values(ascending=False)


# =====================================================================
# 7. EXAMPLE USAGE (mirrors what the original cells did, now as calls)
# =====================================================================
if __name__ == "__main__":
    print(__doc__)
    print("This module is meant to be imported into the notebook, e.g.:")
    print("    from model_comparison_utils import *")
