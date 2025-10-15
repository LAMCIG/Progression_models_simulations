# subtype_clustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def run_clustering(features,
                   n_clusters=3,
                   method="gmm",
                   use_pca=True,
                   pca_components=10,
                   standardize=True,
                   random_state=75):
    """
    AAAAHHHHHHHH
    """
    X = np.array(features, dtype=float)
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = None
    if use_pca:
        max_pc = min(pca_components, X.shape[0], X.shape[1])
        if max_pc < 2:
            max_pc = min(2, X.shape[1])
        pca = PCA(n_components=max_pc, random_state=random_state)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X

    if method.lower() == "gmm":
        model = GaussianMixture(n_components=n_clusters,
                                covariance_type="diag",
                                random_state=random_state)
        model.fit(X_pca)
        labels = model.predict(X_pca)
        probs = model.predict_proba(X_pca)
    else:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_pca)
        probs = None

    out = {
        "labels": np.asarray(labels),
        "probs": probs,
        "X_pca": X_pca,
        "pca": pca,
        "model": model
    }
    return out

def plot_pca_assignments(X_pca, labels, title="PCA clusters", out_path=None):
    """
    Simple scatter of PC1 vs PC2 colored by cluster labels.
    """
    if X_pca.shape[1] < 2:
        print("Not enough PCA components to plot (need at least 2).")
        return

    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = (labels == lab)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=24, alpha=0.8, label=f"cluster {lab}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()

def cluster_and_plot(features,
                     n_clusters=3,
                     method="gmm",
                     use_pca=True,
                     pca_components=10,
                     standardize=True,
                     random_state=75,
                     out_path=None):
    """
    Convenience wrapper: cluster, then plot PC1 vs PC2.
    Returns the clustering dict.
    """
    res = run_clustering(features=features,
                         n_clusters=n_clusters,
                         method=method,
                         use_pca=use_pca,
                         pca_components=pca_components,
                         standardize=standardize,
                         random_state=random_state)
    plot_pca_assignments(res["X_pca"], res["labels"],
                         title=f"PCA ({method}, K={n_clusters})",
                         out_path=out_path)
    return res