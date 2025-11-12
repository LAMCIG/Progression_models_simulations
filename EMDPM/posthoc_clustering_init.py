"""
Post-hoc clustering initialization for EM algorithm.

This module provides functions to initialize EM algorithm using post-hoc clustering
on patient features. This is kept separate from the main EM implementation as
requested, since it requires accurate beta initialization to work well.

Note: This initialization method works best when beta values are already
initialized accurately (e.g., using clinical scores via fit_mixedlm_beta_from_clinical).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .utils import solve_system


def extract_patient_features(X_obs, dt, ids, beta, K, t_span, n_biomarkers):
    """
    Extract features for each patient that can be used for clustering.
    
    Features include:
    - Patient-specific forcing term estimates (from per-patient fits)
    - Mean biomarker values
    - Beta values
    - Trajectory slopes
    
    Parameters
    ----------
    X_obs : np.ndarray
        Observed biomarker values (n_obs, n_biomarkers)
    dt : np.ndarray
        Time since first visit (n_obs,)
    ids : np.ndarray
        Patient IDs (n_obs,)
    beta : np.ndarray
        Beta values per patient (n_patients,)
    K : np.ndarray
        Connectivity matrix
    t_span : np.ndarray
        Time points for trajectory simulation
    n_biomarkers : int
        Number of biomarkers
    
    Returns
    -------
    np.ndarray
        Feature matrix (n_patients, n_features)
    """
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    pid_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}
    
    features = []
    
    for pid in unique_ids:
        mask = (ids == pid)
        X_i = X_obs[mask, :]  # (n_obs_i, n_biomarkers)
        dt_i = dt[mask]
        beta_i = beta[pid_to_index[pid]]
        t_ij = dt_i + beta_i
        
        # Feature 1: Mean biomarker values
        mean_biomarkers = np.mean(X_i, axis=0)
        
        # Feature 2: Beta value
        beta_feature = np.array([beta_i])
        
        # Feature 3: Trajectory slopes (simple linear regression per biomarker)
        slopes = []
        for j in range(n_biomarkers):
            if len(t_ij) >= 2:
                # Simple linear regression
                coeffs = np.polyfit(t_ij, X_i[:, j], 1)
                slopes.append(coeffs[0])  # slope
            else:
                slopes.append(0.0)
        slopes = np.array(slopes)
        
        # Combine features
        patient_features = np.concatenate([mean_biomarkers, beta_feature, slopes])
        features.append(patient_features)
    
    return np.array(features)


def initialize_clusters_from_clustering(X_obs, dt, ids, beta, K, t_span, 
                                        n_biomarkers, n_subtypes,
                                        method="gmm", use_pca=True, 
                                        pca_components=10, random_state=75):
    """
    Initialize cluster assignments and parameters using post-hoc clustering.
    
    This function:
    1. Extracts features from patients
    2. Performs clustering (GMM or KMeans)
    3. Initializes cluster parameters based on cluster assignments
    
    Parameters
    ----------
    X_obs : np.ndarray
        Observed biomarker values
    dt : np.ndarray
        Time since first visit
    ids : np.ndarray
        Patient IDs
    beta : np.ndarray
        Beta values per patient
    K : np.ndarray
        Connectivity matrix
    t_span : np.ndarray
        Time points for trajectory simulation
    n_biomarkers : int
        Number of biomarkers
    n_subtypes : int
        Number of subtypes/clusters
    method : str
        Clustering method: "gmm" or "kmeans"
    use_pca : bool
        Whether to use PCA before clustering
    pca_components : int
        Number of PCA components
    random_state : int
        Random seed
    
    Returns
    -------
    tuple
        (assignments, cluster_f, cluster_scalar_K)
        assignments : np.ndarray of cluster assignments (n_patients,)
        cluster_f : list of forcing terms per cluster
        cluster_scalar_K : list of scalar_K values per cluster
    """
    # Extract features
    features = extract_patient_features(X_obs, dt, ids, beta, K, t_span, n_biomarkers)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA if requested
    pca = None
    if use_pca:
        max_pc = min(pca_components, features_scaled.shape[0], features_scaled.shape[1])
        if max_pc < 2:
            max_pc = min(2, features_scaled.shape[1])
        pca = PCA(n_components=max_pc, random_state=random_state)
        features_reduced = pca.fit_transform(features_scaled)
    else:
        features_reduced = features_scaled
    
    # Perform clustering
    if method.lower() == "gmm":
        model = GaussianMixture(
            n_components=n_subtypes,
            covariance_type="diag",
            random_state=random_state
        )
        model.fit(features_reduced)
        assignments = model.predict(features_reduced)
    else:  # kmeans
        model = KMeans(
            n_clusters=n_subtypes,
            n_init=10,
            random_state=random_state
        )
        assignments = model.fit_predict(features_reduced)
    
    # Initialize cluster parameters based on assignments
    unique_ids = np.unique(ids)
    cluster_f = []
    cluster_scalar_K = []
    
    for subtype in range(n_subtypes):
        # Get patients in this cluster
        cluster_mask = (assignments == subtype)
        cluster_patient_indices = np.where(cluster_mask)[0]
        
        if len(cluster_patient_indices) == 0:
            # Empty cluster - use default initialization
            cluster_f.append(np.random.uniform(0, 0.1, size=n_biomarkers))
            cluster_scalar_K.append(1.0)
            continue
        
        # Get observations for patients in this cluster
        cluster_patient_ids = unique_ids[cluster_patient_indices]
        cluster_patient_mask = np.isin(ids, cluster_patient_ids)
        X_cluster = X_obs[cluster_patient_mask, :]
        dt_cluster = dt[cluster_patient_mask]
        ids_cluster = ids[cluster_patient_mask]
        beta_cluster = beta[cluster_patient_indices]
        
        # Initialize f and scalar_K for this cluster
        # Simple initialization: use mean of observations
        # In practice, you might want to fit a simple model per cluster
        mean_obs = np.mean(X_cluster, axis=0)
        f_init = np.clip(mean_obs * 0.1, 0, 0.1)  # Scale down
        scalar_K_init = float(np.max(X_cluster))
        
        cluster_f.append(f_init)
        cluster_scalar_K.append(scalar_K_init)
    
    return assignments, cluster_f, cluster_scalar_K

