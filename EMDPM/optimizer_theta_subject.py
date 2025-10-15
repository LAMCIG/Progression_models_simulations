import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system

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
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    # print("Breakpoint 1 Theta: ", n_biomarkers, x0.shape, f.shape, K.shape, t_span.shape)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled  = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs) # (n_obs, n_biomarkers)
    # print("Breakpoint 2 Theta: ", x_pred.shape, t_obs.shape, t_span.shape, x_scaled.shape)
    for j in range(n_biomarkers):
        x_pred[:,j] = np.interp(t_obs, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * scalar_K**2
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
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    # print("Breakpoint 1 Theta: ", n_biomarkers, x0.shape, f.shape, K.shape, t_span.shape)
    x = solve_system(x0, f, K, t_span, scalar_K)

    x_scaled  = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs) # (n_obs, n_biomarkers)
    # print("Breakpoint 2 Theta: ", x_pred.shape, t_obs.shape, t_span.shape, x_scaled.shape)
    for j in range(n_biomarkers):
        x_pred[:,j] = np.interp(t_obs, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * (scalar_K**2)

    ### gradient computations
    
    ## wrt f
    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])

    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=True)
        df_obs[:,i] = cs_integ(t_obs)
    
    grad_f = -2 * np.sum(residuals * (df_obs * s[None, :]), axis=0) + np.sign(f) * lambda_f
    
    ## wrt s_k
    Kx_plus_f = (K @ x) + f[:, None]  # n_biomarkers x len(t_span)
    scalar_expr = (1 - x) * Kx_plus_f

    scalar_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        interp_fn = CubicSpline(t_span, scalar_expr[i], extrapolate=True)
        scalar_obs[:,i] = interp_fn(t_obs)

    # grad_scalar_K = -2 * np.sum(residuals * scalar_obs) + 2 * scalar_K
    grad_scalar_K = -2 * np.sum(residuals * scalar_obs) - 2 * scalar_K # 9/10/2025
    # grad_scalar_K = -2 * np.sum(residuals * scalar_obs) + 2 * lambda_scalar * scalar_K
    grad_s = -2 * np.sum(residuals * x_pred, axis=0) # supremum

    # print(grad_f.shape, grad_s.shape, grad_scalar_K.shape)
    # print(grad_f, grad_s, grad_scalar_K)
    grad = np.concatenate([grad_f, grad_s, [grad_scalar_K]])
    return loss, grad

def fit_theta(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
              t_span: np.ndarray, use_jacobian: bool, 
              lambda_f: float, lambda_scalar: float,
              beta_pred: np.ndarray = None, 
              f_guess: np.ndarray = None, scalar_K_guess: float = None, s_guess: np.ndarray = None,
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
def fit_theta_subject(X_obs_i,
                      dt_i,
                      beta_i,
                      K,
                      t_span,
                      use_jacobian,
                      lambda_f,
                      lambda_scalar,
                      f_init=None,
                      s_init=None,
                      scalar_K_init=None,
                      rng=None,
                      bounds_scale=1.0):
    """
    Optimize theta for a single patient with fixed beta.
    Returns f_fit, s_fit, scalar_K_fit.
    """
    if rng is None:
        rng = np.random.default_rng(75)

    n_biomarkers = X_obs_i.shape[1]
    x0_fixed = np.zeros(n_biomarkers)

    # build observation times with fixed beta
    t_obs = dt_i + beta_i

    # initial guesses
    if f_init is None:
        f_init = rng.uniform(0.0, 0.2, size=n_biomarkers)
    if s_init is None:
        s_init = np.ones(n_biomarkers)
    if scalar_K_init is None:
        scalar_K_init = 1.0

    initial_params = np.concatenate([f_init, s_init, [scalar_K_init]])

    # bounds
    bounds_f = [(0.0, np.inf)] * n_biomarkers
    bounds_s = [(0.0, np.inf)] * n_biomarkers
    bounds_scalar_K = [(0.0, np.inf)]
    bounds = bounds_f + bounds_s + bounds_scalar_K

    if use_jacobian:
        loss_fn = theta_loss_jac
        jac_flag = True
    else:
        loss_fn = theta_loss
        jac_flag = False

    result = minimize(
        loss_fn,
        initial_params,
        args=(t_obs, X_obs_i, K, t_span, lambda_f, lambda_scalar),
        method="L-BFGS-B",
        jac=jac_flag,
        bounds=bounds
    )

    fitted = result.x
    f_fit = fitted[:n_biomarkers]
    s_fit = fitted[n_biomarkers:2*n_biomarkers]
    scalar_K_fit = fitted[-1]

    return f_fit, s_fit, scalar_K_fit
