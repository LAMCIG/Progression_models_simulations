import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .utils import solve_system
from .kernel_jsd import KernelJSD
def _jsd_loss_and_grad(beta_i: float, beta_all: np.ndarray, assignments: np.ndarray, 
                       patient_idx: int, lambda_jsd: float, t_max: float) -> tuple:

    if lambda_jsd == 0.0 or len(np.unique(assignments)) != 2:
        return 0.0, 0.0
    
    # Create temporary beta array with updated value for this patient
    beta_temp = beta_all.copy()
    beta_temp[patient_idx] = beta_i
    
    # Extract betas for each subtype
    subtype_0_betas = beta_temp[assignments == 0]
    subtype_1_betas = beta_temp[assignments == 1]
    
    if len(subtype_0_betas) == 0 or len(subtype_1_betas) == 0:
        return 0.0, 0.0
    
    # Compute JSD and derivatives
    jsd_calc = KernelJSD(
        alpha=subtype_0_betas,
        beta=subtype_1_betas,
        value_range=(0, t_max)
    )
    
    # JSD loss: we want to MAXIMIZE JSD, so minimize -JSD
    jsd_value = jsd_calc.jsd()
    jsd_loss = -lambda_jsd * jsd_value
    
    # Get derivatives
    d_alpha, d_beta = jsd_calc.jsd_derivatives()
    
    # Find which derivative corresponds to this patient
    patient_subtype = assignments[patient_idx]
    if patient_subtype == 0:
        # Find index of this patient in subtype 0 array
        subtype_0_indices = np.where(assignments == 0)[0]
        local_idx = np.where(subtype_0_indices == patient_idx)[0][0]
        jsd_grad = -lambda_jsd * d_alpha[local_idx]
    else:  # patient_subtype == 1
        # Find index of this patient in subtype 1 array
        subtype_1_indices = np.where(assignments == 1)[0]
        local_idx = np.where(subtype_1_indices == patient_idx)[0][0]
        jsd_grad = -lambda_jsd * d_beta[local_idx]
    
    return jsd_loss, jsd_grad


def beta_loss(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
              X_pred: np.ndarray, t_span: np.ndarray,
              cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
              theta: np.ndarray, lambda_cog: float = 0.0,
              beta_all: np.ndarray = None, assignments: np.ndarray = None,
              patient_idx: int = None, lambda_jsd: float = 0.0, t_max: float = 12.0) -> float:
    
    """
    Computes the loss for optimizing a single patient's beta_i.

    Parameters
    ----------
    TODO: DOCSTRINGS NEEDED ASAP
    Returns
    -------
    float
        Total loss (reconstruction + cognitive prior + JSD regularization)
    """
    n_biomarkers = X_pred.shape[0]         # int
    s = theta[n_biomarkers:2*n_biomarkers] # (n_biomarkers)
    
    t_pred_i = dt_i + beta_i
    
    X_interp_i = np.array([
        np.interp(t_pred_i, t_span, s[b] * X_pred[b]) # applying supremum
        for b in range(X_pred.shape[0])
    ])
    
    X_obs_i_T = X_obs_i.T  # now (n_biomarkers, n_obs_i)
    residuals = X_obs_i_T - X_interp_i
    
    loss = np.sum(residuals ** 2)

    cog_pred = cog_i @ cog_a + cog_b  # shape (n_visits_i,)
    cog_prior = lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)
    
    # Add JSD regularization if provided
    jsd_loss = 0.0
    if lambda_jsd > 0 and beta_all is not None and assignments is not None and patient_idx is not None:
        jsd_loss, _ = _jsd_loss_and_grad(beta_i, beta_all, assignments, patient_idx, lambda_jsd, t_max)

    return loss + cog_prior + jsd_loss
    
    
def beta_loss_jac(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
                  X_pred: np.ndarray, t_span: np.ndarray,
                  cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
                  theta: np.ndarray, K: np.ndarray, lambda_cog: float = 0.0,
                  beta_all: np.ndarray = None, assignments: np.ndarray = None,
                  patient_idx: int = None, lambda_jsd: float = 0.0, t_max: float = 12.0) -> tuple:
    """
    Computes the loss and gradient for optimizing a single patient's beta_i.

    Parameters
    ----------
    beta_i: float
        Current beta value for patient being optimized
    X_obs_i: (n_obs_i, n_biomarkers)
        Observations for this patient
    dt_i: (n_obs_i,)
        Time deltas for this patient
    X_pred: (n_biomarkers, len(t_span))
        Predicted trajectories
    t_span: (len(t_span),)
        Time span
    cog_i: (n_obs_i, n_cog_features)
        Cognitive features
    cog_a: (n_cog_features,)
        Cognitive regression coefficients
    cog_b: float
        Cognitive regression intercept
    theta: np.ndarray
        Model parameters (f, s, scalar_K)
    K: np.ndarray
        Connectivity matrix
    lambda_cog: float
        Cognitive regularization strength
    beta_all: np.ndarray, optional
        All beta values for JSD computation
    assignments: np.ndarray, optional
        Subtype assignments for all patients
    patient_idx: int, optional
        Index of patient being optimized
    lambda_jsd: float
        JSD regularization strength
    t_max: float
        Maximum time value
    
    Returns
    -------
    tuple
        (loss, grad) - total loss and gradient with respect to beta_i
    """
    n_biomarkers = X_pred.shape[0]         # int
    f = theta[:n_biomarkers]               # (n_biomarkers,)
    s = theta[n_biomarkers:2*n_biomarkers] # (n_biomarkers,)
    scalar_K = theta[-1]                   # float

    t_pred_i = dt_i + beta_i               # (n_visits_i,)

    X_interp_i = np.array([
        np.interp(t_pred_i, t_span, s[b] * X_pred[b])
        for b in range(X_pred.shape[0])
    ])

    X_obs_i_T = X_obs_i.T                  # (n_biomarkers, n_visits_i)
    residuals = X_obs_i_T - X_interp_i     

    # dx/dt = (I - diag(x)) @ (scalar_K K x + f)
    
    dxdt_interp_i = np.zeros_like(X_interp_i)
    for j in range(X_interp_i.shape[1]):
        x_t = X_interp_i[:, j]
        dxdt_interp_i[:, j] = (np.eye(n_biomarkers) - np.diag(x_t)) @ ((scalar_K * K) @ x_t + f)
    dxdt_interp_i *= s[:, None]
    grad_reconstruction = -2 * np.sum(residuals * dxdt_interp_i, axis=(0, 1))
    
    cog_pred = cog_i @ cog_a + cog_b
    cog_prior = lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)
    grad_cog = 2 * lambda_cog * np.sum(t_pred_i - cog_pred)

    loss = np.sum(residuals ** 2) + cog_prior
    grad = grad_reconstruction + grad_cog
    
    # Add JSD regularization if provided
    if lambda_jsd > 0 and beta_all is not None and assignments is not None and patient_idx is not None:
        jsd_loss, jsd_grad = _jsd_loss_and_grad(beta_i, beta_all, assignments, patient_idx, lambda_jsd, t_max)
        loss += jsd_loss
        grad += jsd_grad
    
    return loss, grad

def estimate_beta_for_patient(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
                              X_pred: np.ndarray, t_span: np.ndarray,
                              cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
                              theta: np.ndarray, K: np.ndarray, lambda_cog: float = 0.0,
                              use_jacobian: bool = False, t_max: float = 12.0,
                              beta_all: np.ndarray = None, assignments: np.ndarray = None,
                              patient_idx: int = None, lambda_jsd: float = 0.0) -> float:
    """
    Estimates the optimal beta_i for a single patient.

    Parameters
    ----------
    beta_i : float
        Initial guess for beta
    X_obs_i : np.ndarray
        Observations for this patient
    dt_i : np.ndarray
        Time deltas for this patient
    X_pred : np.ndarray
        Predicted trajectories
    t_span : np.ndarray
        Time span
    cog_i : np.ndarray
        Cognitive features
    cog_a : np.ndarray
        Cognitive regression coefficients
    cog_b : float
        Cognitive regression intercept
    theta : np.ndarray
        Model parameters
    K : np.ndarray
        Connectivity matrix
    lambda_cog : float
        Cognitive regularization strength
    use_jacobian : bool
        Whether to use gradient information
    t_max : float
        Maximum time value
    beta_all : np.ndarray, optional
        All beta values for JSD computation
    assignments : np.ndarray, optional
        Subtype assignments for all patients
    patient_idx : int, optional
        Index of patient being optimized
    lambda_jsd : float
        JSD regularization strength
    
    Returns
    -------
    float
        Optimized beta_i value for the patient.
    """
    beta_guess = beta_i
    
    if use_jacobian == True:
        loss_function = beta_loss_jac
        args=(X_obs_i, dt_i, X_pred, t_span, cog_i, cog_a, cog_b, theta, K, lambda_cog,
              beta_all, assignments, patient_idx, lambda_jsd, t_max)
    else:
        loss_function = beta_loss
        args=(X_obs_i, dt_i, X_pred, t_span, cog_i, cog_a, cog_b, theta, lambda_cog,
              beta_all, assignments, patient_idx, lambda_jsd, t_max)

    result = minimize(
        loss_function,
        x0=beta_guess,
        args=args,
        jac=use_jacobian,
        bounds=[(0, t_max)],
        method="L-BFGS-B"
    )

    return result.x[0]


def _vectorized_beta_loss_and_grad(beta_all: np.ndarray, X_obs: np.ndarray, dt: np.ndarray,
                                   ids: np.ndarray, cog: np.ndarray, t_span: np.ndarray,
                                   cluster_f: list, cluster_scalar_K: list, s: np.ndarray,
                                   assignments: np.ndarray, K: np.ndarray,
                                   cog_a: np.ndarray, cog_b: float,
                                   lambda_cog: float, lambda_jsd: float, t_max: float,
                                   X_pred_by_cluster: dict = None) -> tuple:
    """
    Vectorized loss and gradient for all patients' betas simultaneously.
    
    Parameters
    ----------
    beta_all : np.ndarray
        Current beta values for all patients (n_patients,)
    X_obs : np.ndarray
        Stacked observations (n_obs_total, n_biomarkers)
    dt : np.ndarray
        Stacked time deltas (n_obs_total,)
    ids : np.ndarray
        Patient ID for each observation (n_obs_total,)
    cog : np.ndarray
        Stacked cognitive features (n_obs_total, n_cog_features)
    t_span : np.ndarray
        Time span for trajectories
    cluster_f : list
        List of f vectors, one per cluster
    cluster_scalar_K : list
        List of scalar_K values, one per cluster
    s : np.ndarray
        Scaling parameters (n_biomarkers,)
    assignments : np.ndarray
        Cluster assignment for each patient (n_patients,)
    K : np.ndarray
        Connectivity matrix
    cog_a : np.ndarray
        Cognitive regression coefficients
    cog_b : float
        Cognitive regression intercept
    lambda_cog : float
        Cognitive regularization strength
    lambda_jsd : float
        JSD regularization strength
    t_max : float
        Maximum time value
    X_pred_by_cluster : dict, optional
        Pre-computed X_pred trajectories by cluster (to avoid recomputing)
    
    Returns
    -------
    tuple
        (total_loss, gradient) where gradient is (n_patients,)
    """
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_biomarkers = X_obs.shape[1]
    
    # Pre-compute X_pred for each cluster if not provided
    if X_pred_by_cluster is None:
        X_pred_by_cluster = {}
        for subtype in range(len(cluster_f)):
            f_cluster = np.ravel(cluster_f[subtype])
            scalar_K_cluster = cluster_scalar_K[subtype]
            X_pred_by_cluster[subtype] = solve_system(
                np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K_cluster
            )
    
    total_loss = 0.0
    gradient = np.zeros(n_patients)
    
    # Process each patient
    for idx, patient_id in enumerate(unique_ids):
        mask = (ids == patient_id)
        X_obs_i = X_obs[mask, :]  # (n_obs_i, n_biomarkers)
        dt_i = dt[mask]  # (n_obs_i,)
        cog_i = cog[mask, :]  # (n_obs_i, n_cog_features)
        beta_i = beta_all[idx]
        
        # Get cluster-specific parameters
        subtype_i = assignments[idx]
        f_cluster_i = np.ravel(cluster_f[subtype_i])
        scalar_K_cluster_i = cluster_scalar_K[subtype_i]
        X_pred_cluster_i = X_pred_by_cluster[subtype_i]
        theta_cluster_i = np.concatenate([f_cluster_i, s, [scalar_K_cluster_i]])
        
        # Compute loss and gradient for this patient (WITHOUT JSD - computed separately)
        loss_i, grad_i = beta_loss_jac(
            beta_i, X_obs_i, dt_i, X_pred_cluster_i, t_span,
            cog_i, cog_a, cog_b, theta_cluster_i, K, lambda_cog,
            beta_all=None, assignments=None, patient_idx=None,
            lambda_jsd=0.0, t_max=t_max
        )
        
        total_loss += loss_i
        gradient[idx] = grad_i
    
    # Add JSD loss (only computed once across all patients)
    if lambda_jsd > 0 and len(np.unique(assignments)) == 2:
        subtype_0_betas = beta_all[assignments == 0]
        subtype_1_betas = beta_all[assignments == 1]
        
        if len(subtype_0_betas) > 0 and len(subtype_1_betas) > 0:
            jsd_calc = KernelJSD(
                alpha=subtype_0_betas,
                beta=subtype_1_betas,
                value_range=(0, t_max)
            )
            jsd_value = jsd_calc.jsd()
            jsd_loss = -lambda_jsd * jsd_value
            
            # Get JSD gradients
            d_alpha, d_beta = jsd_calc.jsd_derivatives()
            
            # Map gradients back to patient indices
            subtype_0_indices = np.where(assignments == 0)[0]
            subtype_1_indices = np.where(assignments == 1)[0]
            
            for local_idx, patient_idx in enumerate(subtype_0_indices):
                gradient[patient_idx] += -lambda_jsd * d_alpha[local_idx]
            
            for local_idx, patient_idx in enumerate(subtype_1_indices):
                gradient[patient_idx] += -lambda_jsd * d_beta[local_idx]
            
            total_loss += jsd_loss
    
    return total_loss, gradient


def estimate_beta(beta_all: np.ndarray, X_obs: np.ndarray, dt: np.ndarray,
                  ids: np.ndarray, cog: np.ndarray, t_span: np.ndarray,
                  cluster_f: list, cluster_scalar_K: list, s: np.ndarray,
                  assignments: np.ndarray, K: np.ndarray,
                  cog_a: np.ndarray, cog_b: float,
                  lambda_cog: float = 0.0, lambda_jsd: float = 0.0,
                  t_max: float = 12.0) -> tuple:
    """
    Vectorized estimation of beta values for all patients simultaneously.
    
    This function optimizes all patients' beta values at once using vectorized operations.
    Each patient may have different numbers of observations and different cluster assignments.
    
    Parameters
    ----------
    beta_all : np.ndarray
        Initial beta values for all patients (n_patients,)
    X_obs : np.ndarray
        Stacked observations (n_obs_total, n_biomarkers)
    dt : np.ndarray
        Stacked time deltas (n_obs_total,)
    ids : np.ndarray
        Patient ID for each observation (n_obs_total,)
    cog : np.ndarray
        Stacked cognitive features (n_obs_total, n_cog_features)
    t_span : np.ndarray
        Time span for trajectories
    cluster_f : list
        List of f vectors, one per cluster. Each is (n_biomarkers,)
    cluster_scalar_K : list
        List of scalar_K values, one per cluster
    s : np.ndarray
        Scaling parameters (n_biomarkers,)
    assignments : np.ndarray
        Cluster assignment for each patient (n_patients,)
    K : np.ndarray
        Connectivity matrix
    cog_a : np.ndarray
        Cognitive regression coefficients (n_cog_features,)
    cog_b : float
        Cognitive regression intercept
    lambda_cog : float
        Cognitive regularization strength
    lambda_jsd : float
        JSD regularization strength
    t_max : float
        Maximum time value
    
    Returns
    -------
    tuple
        (optimized_beta, total_lse) where:
        - optimized_beta: np.ndarray (n_patients,) - optimized beta values
        - total_lse: float - total sum of squared errors (residuals + clinical prior + jsd)
    """
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_biomarkers = X_obs.shape[1]
    
    # Pre-compute X_pred trajectories for each cluster
    X_pred_by_cluster = {}
    for subtype in range(len(cluster_f)):
        f_cluster = np.ravel(cluster_f[subtype])
        scalar_K_cluster = cluster_scalar_K[subtype]
        X_pred_by_cluster[subtype] = solve_system(
            np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K_cluster
        )

    def loss_func(beta_vec):
        loss, _ = _vectorized_beta_loss_and_grad(
            beta_vec, X_obs, dt, ids, cog, t_span,
            cluster_f, cluster_scalar_K, s, assignments, K,
            cog_a, cog_b, lambda_cog, lambda_jsd, t_max,
            X_pred_by_cluster
        )
        return loss
    
    def grad_func(beta_vec):
        _, grad = _vectorized_beta_loss_and_grad(
            beta_vec, X_obs, dt, ids, cog, t_span,
            cluster_f, cluster_scalar_K, s, assignments, K,
            cog_a, cog_b, lambda_cog, lambda_jsd, t_max,
            X_pred_by_cluster
        )
        return grad
    
    # optimize all betas simultaneously
    result = minimize(
        loss_func,
        x0=beta_all.copy(),
        method="L-BFGS-B",
        jac=grad_func,
        bounds=[(0, t_max)] * n_patients,
        options={'maxiter': 100}
    )
    
    optimized_beta = result.x
    
    # compute final LSE (residuals + clinical prior + jsd)
    total_lse, _ = _vectorized_beta_loss_and_grad(
        optimized_beta, X_obs, dt, ids, cog, t_span,
        cluster_f, cluster_scalar_K, s, assignments, K,
        cog_a, cog_b, lambda_cog, lambda_jsd, t_max,
        X_pred_by_cluster
    )
    
    return optimized_beta, total_lse
