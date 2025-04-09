import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .utils import solve_system
def beta_loss(beta_i: float, dt_obs: np.ndarray, x_obs: np.ndarray,
              x_reconstructed: np.ndarray, t_span: np.ndarray) -> tuple:
    """
    Computes the loss and gradient for optimizing a single patient's beta_i.

    Parameters
    ----------
    beta_i : float
        Current estimate of the patient-specific time shift.
    dt_obs : np.ndarray
        Relative time since first visit for each observation.
    x_obs : np.ndarray
        Observed biomarker values (n_biomarkers x n_visits).
    x_reconstructed : np.ndarray
        Model-predicted biomarker trajectories from solve_system.
    t_span : np.ndarray
        Time span used to simulate trajectories.

    Returns
    -------
    tuple
        Residual sum of squares and gradient with respect to beta_i.
    """
    t_adjusted = dt_obs + beta_i

    x_pred = np.array([
        np.interp(t_adjusted, t_span, x_reconstructed[i])
        for i in range(x_obs.shape[0])
    ])

    residuals = x_obs - x_pred
    loss = np.sum(residuals ** 2)

    return loss

def beta_loss_jac(beta_i: float, dt_obs: np.ndarray, x_obs: np.ndarray,
              x_reconstructed: np.ndarray, t_span: np.ndarray) -> tuple:
    """
    Computes the loss and gradient for optimizing a single patient's beta_i.

    Parameters
    ----------
    beta_i : float
        Current estimate of the patient-specific time shift.
    dt_obs : np.ndarray
        Relative time since first visit for each observation.
    x_obs : np.ndarray
        Observed biomarker values (n_biomarkers x n_visits).
    x_reconstructed : np.ndarray
        Model-predicted biomarker trajectories from solve_system.
    t_span : np.ndarray
        Time span used to simulate trajectories.

    Returns
    -------
    tuple
        Residual sum of squares and gradient with respect to beta_i.
    """
    t_adjusted = dt_obs + beta_i

    x_pred = np.array([
        np.interp(t_adjusted, t_span, x_reconstructed[i])
        for i in range(x_obs.shape[0])
    ])

    residuals = x_obs - x_pred
    loss = np.sum(residuals ** 2)
    
    

    df_dt = np.array([
        np.gradient(x_reconstructed[i], t_span)
        for i in range(x_obs.shape[0])
    ])
    df_dt_interp = np.array([
        np.interp(t_adjusted, t_span, df_dt[i])
        for i in range(x_obs.shape[0])
    ])

    grad = 2 * np.sum((x_pred - x_obs) * df_dt_interp)
    return loss, grad

def estimate_beta_for_patient(df_patient: pd.DataFrame, x_reconstructed: np.ndarray,
                               t_span: np.ndarray, t_max: float, use_jacobian: bool = False) -> float:
    """
    Estimates the optimal beta_i for a single patient.

    Parameters
    ----------
    df_patient : pd.DataFrame
        Subset of data for a single patient.
    x_reconstructed : np.ndarray
        Current model prediction of biomarker trajectories.
    t_span : np.ndarray
        Time points used in model simulation.
    t_max : float
        Maximum allowed time shift.

    Returns
    -------
    float
        Optimized beta_i value for the patient.
    """
    dt_obs = df_patient["dt"].values
    x_obs = df_patient[[col for col in df_patient.columns if "biomarker_" in col]].values.T

    beta_guess = np.median(dt_obs)
    
    if use_jacobian == True:
        loss_function = beta_loss_jac
    else:
        loss_function = beta_loss

    result = minimize(
        loss_function,
        x0=[beta_guess],
        args=(dt_obs, x_obs, x_reconstructed, t_span),
        jac=use_jacobian,
        bounds=[(0, t_max)],
        method="L-BFGS-B"
    )

    return result.x[0]
