import numpy as np
from scipy.optimize import minimize, LinearConstraint
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
    s0 = params[2*n_biomarkers:3*n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    # print("Breakpoint 1 Theta: ", n_biomarkers, x0.shape, f.shape, K.shape, t_span.shape)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled  = s[:, None] * x + s0[:, None]
    
    x_pred = np.zeros_like(x_obs) # (n_obs, n_biomarkers)
    # print("Breakpoint 2 Theta: ", x_pred.shape, t_obs.shape, t_span.shape, x_scaled.shape)
    for j in range(n_biomarkers):
        x_pred[:,j] = np.interp(t_obs, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * np.sum(np.abs(f)) + lambda_scalar * scalar_K**2

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
    s0 = params[2*n_biomarkers:3*n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    # print("Breakpoint 1 Theta: ", n_biomarkers, x0.shape, f.shape, K.shape, t_span.shape)
    x = solve_system(x0, f, K, t_span, scalar_K)

    x_scaled  = s[:, None] * x + s0[:, None]
    
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

    grad_scalar_K = -2 * np.sum(residuals * scalar_obs) + 2 * lambda_scalar * scalar_K
    grad_s = -2 * np.sum(residuals * x_pred, axis=0) # supremum
    grad_s0 = -2 * np.sum(residuals, axis = 0)
    # print(grad_f.shape, grad_s.shape, grad_scalar_K.shape)
    # print(grad_f, grad_s, grad_scalar_K)
    grad = np.concatenate([grad_f, grad_s, grad_s0, [grad_scalar_K]])
    return loss, grad

def fit_theta(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
              t_span: np.ndarray, use_jacobian: bool, 
              lambda_f: float, lambda_scalar: float,
              beta_pred: np.ndarray = None, 
              f_guess: np.ndarray = None, scalar_K_guess: float = None, s_guess: np.ndarray = None, s0_guess: np.ndarray = None,
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
    
    #print("t_pred: ", t_pred.shape)
    # t_pred = dt_obs + beta_pred[ids]
    
    # fixed vars
    n_biomarkers = X_obs.shape[1]
    x0_fixed = np.zeros(n_biomarkers)
    
    # inital guesses if None
    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    
    initial_params = np.concatenate([f_guess, s_guess, s0_guess, [scalar_K_guess]])
    
    # bounds
    bounds_f = [(0.0, np.inf)] * n_biomarkers
    bounds_s = [(-np.inf, np.inf)] * n_biomarkers  # supremum scaling
    bounds_s0 = [(-np.inf, np.inf)] * n_biomarkers
    bounds_scalar_K = [(0.0, np.inf)] 

    bounds = bounds_f + bounds_s + bounds_s0 + bounds_scalar_K
     
    if use_jacobian == True:
        loss_function = theta_loss_jac
    else:
        loss_function = theta_loss
    
    y_max = np.max(X_obs, axis = 0)
    y_min = np.min(X_obs, axis = 0)
    
    #print("y_max:" ,y_max.shape)
    #print("y_min:" ,y_min.shape)
    #print(y_max)
    
    n_total = 3 * n_biomarkers + 1

    A_top = np.zeros((n_biomarkers, n_total))
    A_top[np.arange(n_biomarkers), n_biomarkers + np.arange(n_biomarkers)] = y_max  # s_k col
    A_top[np.arange(n_biomarkers), 2*n_biomarkers + np.arange(n_biomarkers)] = 1.0  # s0_k col

    A_bot = np.zeros((n_biomarkers, n_total))
    A_bot[np.arange(n_biomarkers), n_biomarkers + np.arange(n_biomarkers)] = y_min
    A_bot[np.arange(n_biomarkers), 2*n_biomarkers + np.arange(n_biomarkers)] = 1.0

    A = np.vstack([A_top, A_bot])
    lc = LinearConstraint(A=A, lb=0.0, ub=1.0)
    
    # = [{'type': 'ineq', 'fun': constr_func}]
    # TODO: rewrite using [{}] syntax
    
    result = minimize(
        loss_function,
        initial_params,
        args=(t_pred, X_obs, K, t_span, lambda_f, lambda_scalar),
        #method="SLSQP",
        method="L-BFGS-B",
        #constraints=lc,
        jac=use_jacobian,
        bounds=bounds
    )

    fitted_params = result.x
    f_fit = fitted_params[:n_biomarkers]
    s_fit = fitted_params[n_biomarkers:2*n_biomarkers]
    s0_fit = fitted_params[2*n_biomarkers:3*n_biomarkers]
    scalar_K_fit = fitted_params[-1]
    
    return x0_fixed, f_fit, s_fit, s0_fit, scalar_K_fit
