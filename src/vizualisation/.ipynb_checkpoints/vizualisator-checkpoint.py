from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_umap_projection(
    latents,
    labels,
    path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Project latent codes to 2D (UMAP with PCA fallback) and save a scatter chart."""

    latents_np = _to_numpy(latents)
    labels_np = _to_numpy(labels)

    try:
        import umap

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean")
        embedding = reducer.fit_transform(latents_np)
        title = "UMAP projection"
    except Exception:
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(latents_np)
        title = "PCA projection"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="tab10", s=8, alpha=0.8)
    plt.title(title)
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_interactive_projection(
    latents,
    labels,
    path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    fallback_image: str | None = None,
) -> None:
    """Create an interactive HTML projection (Plotly if available)."""

    latents_np = _to_numpy(latents)
    labels_np = _to_numpy(labels)

    import plotly.express as px  # type: ignore
    import umap  # type: ignore
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean")
    embedding = reducer.fit_transform(latents_np)
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=labels_np.astype(str),
        title="Interactive UMAP projection",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path)

