import numpy as np
import pandas as pd
from random import seed
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system

def theta_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, step: float = 0.1, t_span: np.ndarray = None,
               lamda: float = 1, alpha: float = 1) -> tuple:
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
    x0 = np.zeros(n_biomarkers)
    f = params
    sparsity_penalty = lamda * np.sum(np.abs(f))

    x = solve_system(x0, f, K, t_span, alpha)

    x_pred = np.zeros((n_biomarkers, len(t_obs)))
    for j in range(n_biomarkers):
        x_pred[j] = np.interp(t_obs, t_span, x[j])

    residuals = x_obs.flatten() - x_pred.flatten()
    loss = np.sum(residuals**2) + sparsity_penalty

    return loss

def theta_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
               K: np.ndarray, step: float = 0.1, t_span: np.ndarray = None,
               lamda: float = 0.01, alpha: float = 1) -> tuple:
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
    x0 = np.zeros(n_biomarkers)
    f = params
    sparsity_penalty = lamda * np.sum(np.abs(f))

    x = solve_system(x0, f, K, t_span, alpha)

    x_pred = np.zeros((n_biomarkers, len(t_obs)))
    for j in range(n_biomarkers):
        x_pred[j] = np.interp(t_obs, t_span, x[j])

    residuals = x_obs.flatten() - x_pred.flatten()
    loss = np.sum(residuals**2) + sparsity_penalty

    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])

    f_obs = np.zeros((n_biomarkers, len(t_obs)))
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=True)
        f_obs[i] = cs_integ(t_obs)

    grad_f = -2 * np.sum(residuals.reshape(n_biomarkers, -1) * f_obs, axis=1) + np.sign(f) * lamda 

    return loss, grad_f

def fit_theta(df_opt: pd.DataFrame, beta_iter: pd.DataFrame,
              iteration: int, K: np.ndarray,
              t_span: np.ndarray, step: float = 0.1,
              use_jacobian: bool = False, lamda: float = 1,
              alpha: float = 1.0) -> tuple:
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
    
    np.random.seed(75)
    seed(75)
    
    t_obs = df_opt["dt"].values + beta_iter[str(iteration)].values
    x_obs = df_opt[[col for col in df_opt.columns if "biomarker_" in col]].values.T

    n_biomarkers = x_obs.shape[0]
    x0_fixed = np.zeros(n_biomarkers)
    f_min, f_max = 0, 0.2
    f_guess = np.random.uniform(f_min, f_max, size=n_biomarkers)
    bounds = [(f_min, f_max)] * n_biomarkers
    
    if use_jacobian == True:
        loss_function = theta_loss_jac
    else:
        loss_function = theta_loss

    result = minimize(
        loss_function,
        f_guess,
        args=(t_obs, x_obs, K, step, t_span, lamda, alpha),
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )

    f_fit = result.x
    return x0_fixed, f_fit
