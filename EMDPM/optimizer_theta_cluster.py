import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system

def theta_cluster_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                       K: np.ndarray, t_span: np.ndarray, s: np.ndarray,
                       lambda_f: float, lambda_scalar: float) -> float:
    """
    Computes loss for optimizing cluster-level f and scalar_K (with fixed global s).
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [f, scalar_K] where f is (n_biomarkers,) and scalar_K is scalar.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    s : np.ndarray
        Fixed global scaling parameter (shape: n_biomarkers).
    lambda_f : float
        Regularization strength for f.
    lambda_scalar : float
        Regularization strength for scalar_K.
        
    Returns
    -------
    float
        Loss value.
    """
    n_biomarkers = x_obs.shape[1]
    
    f = params[:n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * scalar_K**2
    
    return loss

def theta_cluster_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                           K: np.ndarray, t_span: np.ndarray, s: np.ndarray,
                           lambda_f: float, lambda_scalar: float) -> tuple:
    """
    Computes loss and gradient for optimizing cluster-level f and scalar_K.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [f, scalar_K] where f is (n_biomarkers,) and scalar_K is scalar.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    s : np.ndarray
        Fixed global scaling parameter (shape: n_biomarkers).
    lambda_f : float
        Regularization strength for f.
    lambda_scalar : float
        Regularization strength for scalar_K.
        
    Returns
    -------
    tuple
        Loss scalar and gradient vector [grad_f, grad_scalar_K].
    """
    n_biomarkers = x_obs.shape[1]
    
    # Unpack parameters
    f = params[:n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * scalar_K**2
    
    ### Gradient computations
    
    ## Gradient with respect to f
    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])
    
    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=True)
        df_obs[:, i] = cs_integ(t_obs)
    
    # Need to account for s scaling: d/ds(s*x) with respect to f
    grad_f = -2 * np.sum(residuals * (df_obs * s[None, :]), axis=0) + np.sign(f) * lambda_f
    
    ## Gradient with respect to scalar_K
    Kx_plus_f = (K @ x) + f[:, None]  # n_biomarkers x len(t_span)
    scalar_expr = (1 - x) * Kx_plus_f
    
    scalar_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        interp_fn = CubicSpline(t_span, scalar_expr[i], extrapolate=True)
        scalar_obs[:, i] = interp_fn(t_obs)
    
    # Need to account for s scaling: d/ds(s*x) with respect to scalar_K
    grad_scalar_K = -2 * np.sum(residuals * (scalar_obs * s[None, :])) + 2 * lambda_scalar * scalar_K
    
    grad = np.concatenate([grad_f, [grad_scalar_K]])
    return loss, grad

def fit_theta_cluster(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, use_jacobian: bool,
                      s: np.ndarray,
                      lambda_f: float, lambda_scalar: float,
                      beta_pred: np.ndarray = None,
                      f_guess: np.ndarray = None, scalar_K_guess: float = None,
                      rng: np.random.Generator = None) -> tuple:
    """
    Optimizes cluster-level f and scalar_K for patients in a specific cluster.
    
    Parameters
    ----------
    X_obs : np.ndarray
        Observed biomarker values for cluster patients (shape: n_obs_cluster x n_biomarkers).
    dt_obs : np.ndarray
        Time deltas from baseline (shape: n_obs_cluster).
    ids : np.ndarray
        Patient IDs for each observation (shape: n_obs_cluster).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time span for solving ODE.
    use_jacobian : bool
        Whether to use Jacobian in optimization.
    s : np.ndarray
        Fixed global scaling parameter (shape: n_biomarkers).
    lambda_f : float
        Regularization strength for f.
    lambda_scalar : float
        Regularization strength for scalar_K.
    beta_pred : np.ndarray
        Predicted beta values per patient (shape: n_patients_cluster).
    f_guess : np.ndarray
        Initial guess for f (shape: n_biomarkers).
    scalar_K_guess : float
        Initial guess for scalar_K.
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    tuple
        (f_fit, scalar_K_fit) where f_fit is (n_biomarkers,) and scalar_K_fit is float.
    """
    if rng is None:
        rng = np.random.default_rng(75)
        
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs_cluster,)
    t_pred = dt_obs + beta_pred[index_array]
    
    n_biomarkers = X_obs.shape[1]
    x0_fixed = np.zeros(n_biomarkers)
    
    # Initial guesses if None
    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    
    initial_params = np.concatenate([f_guess, [scalar_K_guess]])
    
    # Bounds
    bounds_f = [(0.0, np.inf)] * n_biomarkers
    bounds_scalar_K = [(0.0, np.inf)]
    bounds = bounds_f + bounds_scalar_K
    
    if use_jacobian:
        loss_function = theta_cluster_loss_jac
    else:
        loss_function = theta_cluster_loss
    
    result = minimize(
        loss_function,
        initial_params,
        args=(t_pred, X_obs, K, t_span, s, lambda_f, lambda_scalar),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )
    
    fitted_params = result.x
    f_fit = fitted_params[:n_biomarkers]
    scalar_K_fit = fitted_params[-1]
    
    return f_fit, scalar_K_fit

