import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from .utils import solve_system

def theta_s_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                 K: np.ndarray, t_span: np.ndarray, f: np.ndarray, scalar_K: float,
                 lambda_s: float = 0.0) -> float:
    """
    Computes loss for optimizing only the global scaling parameter s.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess for s (shape: n_biomarkers).
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    f : np.ndarray
        Fixed forcing term (shape: n_biomarkers).
    scalar_K : float
        Fixed scalar_K parameter.
    lambda_s : float
        Regularization strength for s (optional).
        
    Returns
    -------
    float
        Loss value.
    """
    n_biomarkers = x_obs.shape[1]
    s = params  # Only optimizing s
    
    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2)
    
    return loss

def theta_s_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                     K: np.ndarray, t_span: np.ndarray, f: np.ndarray, scalar_K: float,
                     lambda_s: float = 0.0) -> tuple:
    """
    Computes loss and gradient for optimizing only the global scaling parameter s.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess for s (shape: n_biomarkers).
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    f : np.ndarray
        Fixed forcing term (shape: n_biomarkers).
    scalar_K : float
        Fixed scalar_K parameter.
    lambda_s : float
        Regularization strength for s (optional).
        
    Returns
    -------
    tuple
        Loss scalar and gradient vector.
    """
    n_biomarkers = x_obs.shape[1]
    s = params  # Only optimizing s
    
    x0 = np.zeros(n_biomarkers)
    # print(f.shape, K.shape, t_span.shape, scalar_K)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2)
    
    # Gradient with respect to s
    # d/ds (s * x) = x, so gradient of residual^2 is -2 * residuals * x_pred / s
    # But x_pred = s * x_interp, so we need to use x_interp (unscaled)
    x_interp = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_interp[:, j] = np.interp(t_obs, t_span, x[j])
    
    grad_s = -2 * np.sum(residuals * x_interp, axis=0) + 2 * lambda_s * s
    
    return loss, grad_s

def fit_theta_globals(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, use_jacobian: bool,
                      f: np.ndarray, scalar_K: float,
                      beta_pred: np.ndarray = None,
                      s_guess: np.ndarray = None,
                      lambda_s: float = 0.0,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Optimizes only the global scaling parameter s (supremum) for all patients.
    
    Parameters
    ----------
    X_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    dt_obs : np.ndarray
        Time deltas from baseline (shape: n_obs).
    ids : np.ndarray
        Patient IDs for each observation (shape: n_obs).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time span for solving ODE.
    use_jacobian : bool
        Whether to use Jacobian in optimization.
    f : np.ndarray
        Fixed forcing term (shape: n_biomarkers).
    scalar_K : float
        Fixed scalar_K parameter.
    beta_pred : np.ndarray
        Predicted beta values per patient (shape: n_patients).
    s_guess : np.ndarray
        Initial guess for s (shape: n_biomarkers).
    lambda_s : float
        Regularization strength for s.
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    np.ndarray
        Fitted s values (shape: n_biomarkers).
    """
    if rng is None:
        rng = np.random.default_rng(75)
        
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt_obs + beta_pred[index_array]
    
    n_biomarkers = X_obs.shape[1]
    
    # Initial guess if None
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    
    # Bounds: s must be non-negative
    bounds = [(0.0, np.inf)] * n_biomarkers
    
    if use_jacobian:
        loss_function = theta_s_loss_jac
    else:
        loss_function = theta_s_loss
    
    result = minimize(
        loss_function,
        s_guess,
        args=(t_pred, X_obs, K, t_span, f, scalar_K, lambda_s),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )
    
    s_fit = result.x
    
    return s_fit
