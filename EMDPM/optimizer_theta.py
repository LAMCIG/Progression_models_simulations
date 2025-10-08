import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system, compute_sK_sens, compute_kappa_sens

def theta_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, t_span: np.ndarray, lambda_f: float, lambda_scalar: float) -> tuple:
    """
    Computes residual loss and gradient for estimating the ODE forcing term f.

    Parameters
    ----------
    params : np.ndarray
        Current guess for the forcing term f.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_biomarkers x n_obs).
    K : np.ndarray
        Connectivity matrix.
    step : float, optional
        Time step for integration.
    t_span : np.ndarray
        Time points for trajectory simulation.
    lambda_f : float, optional
        Regularization strength.

    Returns
    -------
    tuple
        Loss scalar and gradient vector.
    """
    n_biomarkers = x_obs.shape[1]

    f = params[:n_biomarkers]
    s = params[n_biomarkers:2*n_biomarkers]
    scalar_K = params[-2]
    kappa = params[-1]
    
    x0 = np.zeros(n_biomarkers)
    x_traj = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled  = s[:, None] * x_traj
    
    x_pred = np.zeros_like(x_obs) # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:,j] = np.interp(t_obs, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * (scalar_K**2 + kappa**2)
    # loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * scalar_K**2
    # loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + scalar_K**2

    return loss
   
def theta_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, t_span: np.ndarray, lambda_f: float, lambda_scalar: float) -> tuple:
    """
    Computes residual loss and gradient for estimating the ODE forcing term f.

    Parameters
    ----------
    params : np.ndarray
        Current guess for the forcing term f.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_biomarkers x n_obs).
    K : np.ndarray
        Connectivity matrix.
    step : float, optional
        Time step for integration.
    t_span : np.ndarray
        Time points for trajectory simulation.
    lambda_f : float, optional
        Regularization strength.

    Returns
    -------
    tuple
        Loss scalar and gradient vector.
    """
    n_biomarkers = x_obs.shape[1]
    
    # unpack parameters
    f = params[:n_biomarkers]
    s = params[n_biomarkers:2*n_biomarkers]
    scalar_K = params[-2]
    kappa = params[-1]

    x0 = np.zeros(n_biomarkers)
    x_traj = solve_system(x0, f, K, t_span, scalar_K, kappa)
    # x_traj = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled  = s[:, None] * x_traj

    # predict at observation times
    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * (scalar_K**2 + kappa**2)

    ### GRADIENTS ###
    
    ## wrt f: forcing term
    cum_int = np.array([
        cumulative_simpson(1 - x_traj[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])
    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=True)
        df_obs[:, i] = cs_integ(t_obs)
    grad_f = -2.0 * np.sum(residuals * (df_obs * s[None, :]), axis=0) + np.sign(f) * lambda_f
        
    # wrt s: x at obs
    x_at_obs = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_at_obs[:, j] = np.interp(t_obs, t_span, x_traj[j])
    grad_s = -2.0 * np.sum(residuals * x_at_obs, axis=0)

    # wrt scalar_K
    t_grid, x_grid_sens, S_alpha = compute_sK_sens(x0, f, K, t_span, scalar_K, kappa, t_eval=t_span)
    S_alpha_obs = np.column_stack([np.interp(t_obs, t_grid, S_alpha[k]) for k in range(n_biomarkers)])
    grad_scalar_K = -2.0 * np.sum(residuals * (S_alpha_obs * s[None, :])) + 2.0 * lambda_scalar * scalar_K

    # wrt kappa
    t_grid2, x_grid_sens2, S_kappa = compute_kappa_sens(x0, f, K, t_span, scalar_K, kappa, t_eval=t_span)
    S_kappa_obs = np.column_stack([np.interp(t_obs, t_grid2, S_kappa[k]) for k in range(n_biomarkers)])
    grad_kappa = -2.0 * np.sum(residuals * (S_kappa_obs * s[None, :])) + 2.0 * lambda_scalar * kappa

    grad = np.concatenate([grad_f, grad_s, [grad_scalar_K, grad_kappa]])
    return loss, grad
    

def fit_theta(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
              t_span: np.ndarray, use_jacobian: bool,
              lambda_f: float, lambda_scalar: float,
              beta_pred: np.ndarray = None,
              f_guess: np.ndarray = None, scalar_K_guess: float = None, s_guess: np.ndarray = None,
              kappa_guess: float = None,
              rng: np.random.Generator = None) -> tuple:
    """
    Optimizes the ODE forcing term f (theta) for current EM iteration.

    Parameters
    ----------
    df_opt : pd.DataFrame
        Subset of observation data.
    beta_iter : pd.DataFrame
        DataFrame containing beta estimates per patient.
    iteration : int
        Current EM iteration.
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time span for solving ODE.
    step : float, optional
        Integration step size.

    Returns
    -------
    tuple
        x0_fixed (np.ndarray), f_fit (np.ndarray)
    """
    if rng is None:
        rng = np.random.default_rng(75)

    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt_obs + beta_pred[index_array]

    n_biomarkers = X_obs.shape[1]
    x0_fixed = np.zeros(n_biomarkers)

    # initial guesses
    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    if kappa_guess is None:
        kappa_guess = 0.1

    initial_params = np.concatenate([f_guess, s_guess, [scalar_K_guess, kappa_guess]])

    # bounds
    bounds_f = [(0.0, np.inf)] * n_biomarkers
    bounds_s = [(0.0, np.inf)] * n_biomarkers
    bounds_scalar_K = [(0.0, np.inf)]
    bounds_kappa = [(0.0, np.inf)]
    bounds = bounds_f + bounds_s + bounds_scalar_K + bounds_kappa

    loss_function = theta_loss_jac if use_jacobian else theta_loss

    result = minimize(
        loss_function,
        initial_params,
        args=(t_pred, X_obs, K, t_span, lambda_f, lambda_scalar),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )

    fitted_params = result.x
    f_fit = fitted_params[:n_biomarkers]
    s_fit = fitted_params[n_biomarkers:2*n_biomarkers]
    scalar_K_fit = fitted_params[-2]
    kappa_fit = fitted_params[-1]

    return x0_fixed, f_fit, s_fit, scalar_K_fit, kappa_fit
