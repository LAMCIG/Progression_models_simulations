import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .utils import solve_system
def beta_loss(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
              X_pred: np.ndarray, t_span: np.ndarray,
              cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
              lambda_cog: float = 0.0) -> float:
    
    """
    Computes the loss and gradient for optimizing a single patient's beta_i.

    Parameters
    ----------
    TODO: DOCSTRINGS NEEDED ASAP
    Returns
    -------
    tuple
        Residual sum of squares and gradient with respect to beta_i.
    """
    t_pred_i = dt_i + beta_i
    X_interp_i = np.array([
        np.interp(t_pred_i, t_span, X_pred[i])
        for i in range(X_pred.shape[0])
    ])
    
    X_obs_i_T = X_obs_i.T  # now (n_biomarkers, n_visits_i)
    residuals = X_obs_i_T - X_interp_i
    loss = np.sum(residuals ** 2)

    cog_pred = cog_i @ cog_a + cog_b  # shape (n_visits_i,)
    cog_prior = lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)

    return loss + cog_prior
    
    
def beta_loss_jac(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
                  X_pred: np.ndarray, t_span: np.ndarray,
                  cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
                  theta: np.ndarray, K: np.ndarray, lambda_cog: float = 0.0) -> float:
    """
    Computes the loss and gradient for optimizing a single patient's beta_i.

    Parameters
    ----------
    Returns
    -------
    tuple
        Residual sum of squares and gradient with respect to beta_i.
    """
    
    n_biomarkers = X_obs_i.shape[1]  # X_pred.shape[0]
    f = theta[:n_biomarkers]
    s = theta[n_biomarkers:2*n_biomarkers]
    scalar_K = theta[-1]
    
    t_pred_i = dt_i + beta_i 

    X_interp_i = np.array([
        np.interp(t_pred_i, t_span, X_pred[i])
        for i in range(X_pred.shape[0])
    ])

    X_obs_i_T = X_obs_i.T  # (n_biomarkers, n_visits_i)
    residuals = X_obs_i_T - X_interp_i
    
    #print("residual terms: ", X_obs_i_T.shape, X_interp_i.shape)

    # dx/dt = (I - diag(x)) @ (scalar_K K x + f)
    I = np.eye(X_pred.shape[0])
    #print("Eye terms: ", I.shape, X_pred.shape, np.diag(X_interp_i).shape)
    
    dxdt_interp_i = np.zeros_like(X_interp_i)
    for j in range(X_interp_i.shape[1]):
        x_t = X_interp_i[:, j]
        dxdt_interp_i[:, j] = (np.eye(n_biomarkers) - np.diag(x_t)) @ (scalar_K * K @ x_t + f)
        grad_reconstruction = -2 * np.sum(residuals * dxdt_interp_i)

    # print("cog params:", cog_i.shape, cog_a.shape, np.size(cog_b))
    cog_pred = cog_i @ cog_a + cog_b
    cog_prior = lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)
    grad_cog = 2 * lambda_cog * np.sum(t_pred_i - cog_pred)

    loss = np.sum(residuals ** 2) + cog_prior
    grad = grad_reconstruction + grad_cog
    return loss, grad

def estimate_beta_for_patient(beta_i: float, df_patient: pd.DataFrame, x_reconstructed: np.ndarray,
                               t_span: np.ndarray, t_max: float, use_jacobian: bool = False,
                               lambda_cog: float = 0, a: float = 1, b: float = 0) -> float:
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

    #beta_guess = np.median(dt_obs)
    beta_guess = beta_i
    s_ij = df_patient["cognitive_score"].values # np.ndarray of cog scores
    
    if use_jacobian == True:
        loss_function = beta_loss_jac
    else:
        loss_function = beta_loss

    result = minimize(
        loss_function,
        x0=beta_guess,
        args=(dt_obs, x_obs, x_reconstructed, t_span, lambda_cog, s_ij, a, b),
        jac=use_jacobian,
        bounds=[(0, t_max)],
        method="L-BFGS-B"
    )

    return result.x[0]
