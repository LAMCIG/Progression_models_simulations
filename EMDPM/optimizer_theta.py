import numpy as np
import pandas as pd
from random import seed
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system

def theta_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, step: float = 0.1, t_span: np.ndarray = None,
               lamda: float = 1) -> tuple:
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
    lamda : float, optional
        Regularization strength.

    Returns
    -------
    tuple
        Loss scalar and gradient vector.
    """
    n_biomarkers = x_obs.shape[0]

    # unpack parameters
    f = params[:n_biomarkers]
    a = params[n_biomarkers:2*n_biomarkers]
    b = params[2*n_biomarkers:3*n_biomarkers]
    scalar_K = params[-1]

    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_affine = a[:, None] * x + b[:, None] # apply affine transform

    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[j] = np.interp(t_obs, t_span, x_affine[j])

    residuals = x_obs.flatten() - x_pred.flatten()
    loss = np.sum(residuals**2) + lamda * np.sum(np.abs(f))  # Only penalize f for now

    return loss

def theta_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, step: float = 0.1, t_span: np.ndarray = None,
               lamda: float = 0.01) -> tuple:
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
    lamda : float, optional
        Regularization strength.

    Returns
    -------
    tuple
        Loss scalar and gradient vector.
    """
    n_biomarkers = x_obs.shape[0]
    
    # unpack parameters
    f = params[:n_biomarkers]
    a = params[n_biomarkers:2*n_biomarkers]
    b = params[2*n_biomarkers:3*n_biomarkers]
    scalar_K = params[-1]

    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)

    x_affine = a[:, None] * x + b[:, None] # apply affine transform
    
    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[j] = np.interp(t_obs, t_span, x_affine[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lamda * np.sum(np.abs(f))

    ## gradient computations
    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])

    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=True)
        df_obs[i] = cs_integ(t_obs)

    grad_f = -2 * np.sum(residuals * (df_obs * a[:, None]), axis=1) + np.sign(f) * lamda
    grad_a = -2 * np.sum(residuals, axis=1) # wrt to a
    grad_b = -2 * np.sum(residuals, axis=1) # wrt to b

    # Gradient wrt scalar_K: skipped for now
    grad_scalar_K = np.array([0.0])

    grad = np.concatenate([grad_f, grad_a, grad_b, grad_scalar_K])
    return loss, grad

def fit_theta(df_opt: pd.DataFrame, beta_iter: pd.DataFrame,
              iteration: int, K: np.ndarray,
              t_span: np.ndarray, step: float = 0.1,
              use_jacobian: bool = False, lamda: float = 1,
              scalar_K_guess: float = None,
              f_guess: np.ndarray = None,
              a_guess: np.ndarray = None,
              b_guess: np.ndarray = None,
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
    
    t_obs = df_opt["dt"].values + beta_iter[str(iteration)].values
    x_obs = df_opt[[col for col in df_opt.columns if "biomarker_" in col]].values.T

    # fixed vars
    n_biomarkers = x_obs.shape[0]
    x0_fixed = np.zeros(n_biomarkers)
    
    # inital guesses if None
    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)
    if a_guess is None:
        a_guess = np.ones(n_biomarkers)
    if b_guess is None:
        b_guess = np.zeros(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    
    initial_params = np.concatenate([f_guess, a_guess, b_guess, scalar_K_guess])
    
    # bounds
    bounds_f = [(0, 0.2)] * n_biomarkers
    bounds_a = [(0.1, 5.0)] * n_biomarkers  # prevent zero scaling
    bounds_b = [(-2.0, 2.0)] * n_biomarkers  # assuming biomarker range ~[0, 1] for, #TODO: ask bg later
    bounds_scalar_K = [(0.01, 5.0)]

    bounds = bounds_f + bounds_a + bounds_b + bounds_scalar_K
     
    if use_jacobian == True:
        loss_function = theta_loss_jac
    else:
        loss_function = theta_loss

    result = minimize(
        loss_function,
        initial_params,
        args=(t_obs, x_obs, K, step, t_span, lamda),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )

    fitted_params = result.x
    f_fit = fitted_params[:n_biomarkers]
    a_fit = fitted_params[n_biomarkers:2*n_biomarkers]
    b_fit = fitted_params[2*n_biomarkers:3*n_biomarkers]
    scalar_K_fit = fitted_params[-1]
    
    return x0_fixed, f_fit, a_fit, b_fit, scalar_K_fit
