import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from .utils import solve_system

def theta_s_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                 K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                 lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> float:
    """
    Computes loss for optimizing global scaling parameter s and scalar_K.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [s, scalar_K] where s is (n_biomarkers,) and scalar_K is scalar.
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
    lambda_s : float
        Regularization strength for s (optional).
    lambda_scalar : float
        Regularization strength for scalar_K (optional).
        
    Returns
    -------
    float
        Loss value.
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    
    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lambda_scalar * scalar_K**2
    
    return loss

def theta_s_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                     K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                     lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    """
    Computes loss and gradient for optimizing global scaling parameter s and scalar_K.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [s, scalar_K] where s is (n_biomarkers,) and scalar_K is scalar.
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
    lambda_s : float
        Regularization strength for s (optional).
    lambda_scalar : float
        Regularization strength for scalar_K (optional).
        
    Returns
    -------
    tuple
        Loss scalar and gradient vector [grad_s, grad_scalar_K].
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    
    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lambda_scalar * scalar_K**2
    
    # Gradient with respect to s
    # d/ds (s * x) = x, so gradient of residual^2 is -2 * residuals * x_pred / s
    # But x_pred = s * x_interp, so we need to use x_interp (unscaled)
    x_interp = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_interp[:, j] = np.interp(t_obs, t_span, x[j])
    
    grad_s = -2 * np.sum(residuals * x_interp, axis=0) + 2 * lambda_s * s
    
    # Gradient with respect to scalar_K
    # Need to compute d/d(scalar_K) of x
    from scipy.interpolate import CubicSpline
    from scipy.integrate import cumulative_simpson
    
    Kx_plus_f = (K @ x) + f[:, None]  # n_biomarkers x len(t_span)
    scalar_expr = (1 - x) * Kx_plus_f
    
    scalar_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        interp_fn = CubicSpline(t_span, scalar_expr[i], extrapolate=True)
        scalar_obs[:, i] = interp_fn(t_obs)
    
    # Need to account for s scaling: d/ds(s*x) with respect to scalar_K
    grad_scalar_K = -2 * np.sum(residuals * (scalar_obs * s[None, :])) + 2 * lambda_scalar * scalar_K
    
    grad = np.concatenate([grad_s, [grad_scalar_K]])
    
    return loss, grad

def fit_theta_globals(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, use_jacobian: bool,
                      f: np.ndarray,
                      beta_pred: np.ndarray = None,
                      s_guess: np.ndarray = None, scalar_K_guess: float = None,
                      lambda_s: float = 0.0, lambda_scalar: float = 0.0,
                      rng: np.random.Generator = None) -> tuple:
    """
    Optimizes global scaling parameter s (supremum) and scalar_K for all patients.
    
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
    beta_pred : np.ndarray
        Predicted beta values per patient (shape: n_patients).
    s_guess : np.ndarray
        Initial guess for s (shape: n_biomarkers).
    scalar_K_guess : float
        Initial guess for scalar_K.
    lambda_s : float
        Regularization strength for s.
    lambda_scalar : float
        Regularization strength for scalar_K.
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    tuple
        (s_fit, scalar_K_fit) where s_fit is (n_biomarkers,) and scalar_K_fit is float.
    """
    if rng is None:
        rng = np.random.default_rng(75)
        
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt_obs + beta_pred[index_array]
    
    n_biomarkers = X_obs.shape[1]
    
    # Initial guesses if None
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    
    # Combine parameters: [s, scalar_K]
    initial_params = np.concatenate([s_guess, [scalar_K_guess]])
    
    # Bounds: s and scalar_K must be non-negative
    bounds = [(0.0, np.inf)] * n_biomarkers + [(0.0, np.inf)]
    
    if use_jacobian:
        loss_function = theta_s_loss_jac
    else:
        loss_function = theta_s_loss
    
    result = minimize(
        loss_function,
        initial_params,
        args=(t_pred, X_obs, K, t_span, f, lambda_s, lambda_scalar),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )
    
    fitted_params = result.x
    s_fit = fitted_params[:n_biomarkers]
    scalar_K_fit = fitted_params[-1]
    
    return s_fit, scalar_K_fit
