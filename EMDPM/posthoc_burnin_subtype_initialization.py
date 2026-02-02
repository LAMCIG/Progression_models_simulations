import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .subject_EM import SubjectEM


def initialize_subtypes_with_burnin(X: list, n_subtypes: int, K: np.ndarray,
                                    t_max: float = 40.0, step: float = 0.01,
                                    random_state: int = 75, lambda_f: float = 1.0,
                                    lambda_scalar: float = 0.3, verbose: int = 1,
                                    use_sparse_pca: bool = None,
                                    spca_components: int = None,
                                    spca_alpha: float = 0.1) -> list:
    """
    Post-hoc burn-in based subtype initialization. Runs a per-patient SubjectEM "burn-in"
    to estimate per-patient forcing terms f_i. Clusters only on f_i (s is global, not subtype-specific).
    Optionally applies sparse PCA for high-dimensional data. Writes 'initial_subtype' to each patient dict.
    """

    # SubjectEM burn-in to get per-patient f_i
    subject_em = SubjectEM(
        K=K,
        t_max=t_max,
        step=step,
        max_iter=200,
        use_jacobian=True,
        lambda_f=lambda_f,
        lambda_scalar=lambda_scalar,
        verbose=verbose,
    )
    subject_em.fit(X)

    # Build feature matrix: ONLY f_i (s is global, not subtype-specific)
    n_patients = len(X)
    n_biomarkers = np.asarray(X[0]["final_f"]).shape[0]
    features = np.array([np.asarray(p["final_f"]).ravel() for p in X])

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply sparse PCA if needed (auto-enable for 10+ biomarkers)
    if use_sparse_pca is None:
        use_sparse_pca = (n_biomarkers >= 10)
    
    if use_sparse_pca:
        from sklearn.decomposition import SparsePCA
        if spca_components is None:
            spca_components = min(10, n_biomarkers - 1, n_patients - 1)
        
        spca = SparsePCA(
            n_components=spca_components,
            alpha=spca_alpha,
            ridge_alpha=0.01,
            max_iter=1000,
            random_state=random_state
        )
        features_reduced = spca.fit_transform(features_scaled)
        if verbose >= 1:
            print(f"  SparsePCA: {n_biomarkers}D -> {spca_components}D")
    else:
        features_reduced = features_scaled

    # KMeans clustering with balanced initialization
    best_labels = None
    best_balance = np.inf
    
    for _ in range(20):
        kmeans = KMeans(
            n_clusters=n_subtypes,
            n_init=1,
            random_state=random_state if _ == 0 else None,
        )
        labels = kmeans.fit_predict(features_reduced)
        
        # Select most balanced clustering
        unique, counts = np.unique(labels, return_counts=True)
        balance_score = np.max(counts) - np.min(counts)
        if balance_score < best_balance:
            best_balance = balance_score
            best_labels = labels

    # Write initial_subtype to patient dicts
    for i, p in enumerate(X):
        p["initial_subtype"] = int(best_labels[i])

    if verbose >= 1:
        unique, counts = np.unique(best_labels, return_counts=True)
        print(f"  Burn-in init: cluster sizes {dict(zip(unique, counts))}")

    return X

