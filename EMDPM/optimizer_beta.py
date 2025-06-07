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
    #print(type(theta), len(theta), theta)
    f = theta[1]
    #print(f.size)
    s = theta[2]
    scalar_K = theta[-1]
    
    t_pred_i = dt_i + beta_i 

    X_interp_i = np.stack([
        np.interp(t_pred_i, t_span, X_pred[i])
        for i in range(X_pred.shape[0])
    ])

    X_obs_i_T = X_obs_i.T  # (n_biomarkers, n_visits_i)
    residuals = X_obs_i_T - X_interp_i
    
    #print("residual terms: ", X_obs_i_T.shape, X_interp_i.shape)

    # dx/dt = (I - diag(x)) @ (scalar_K K x + f)
    I = np.eye(X_pred.shape[0])
    
    # print("Eye terms: ", I.shape, X_pred.shape, np.diag(X_interp_i).shape)
    
    dxdt_interp_i = np.zeros_like(X_interp_i)
    for j in range(X_interp_i.shape[1]):
        x_t = X_interp_i[:, j]
        
        assert x_t.ndim == 1 and x_t.shape[0] == n_biomarkers
        assert np.issubdtype(x_t.dtype, np.floating)
        
        #print(type(f), f.shape)
        
        # print(scalar_K)
        # print(K.shape)
        # print(x_t.shape)
        #print(f)
        #print(type(f))
        #print(f.shape) # find out why the type changes
        
        conn_term = np.array(scalar_K * K @ x_t + np.array(f))
        
        #print("breakpoint 5: ", x_t.shape, (K @ x_t).shape, np.diag(x_t).shape, np.size(((np.eye(n_biomarkers) - np.diag(x_t)) @ (scalar_K * K @ x_t + f))))
        #y = (np.eye(n_biomarkers) - np.diag(x_t)) @ (scalar_K * K @ x_t + f)
        #print("y shape", y.shape, "y dtype", y.dtype, "y contents", y)
        
        dxdt_interp_i[:, j] = (np.eye(n_biomarkers) - np.diag(x_t)) @ conn_term
        grad_reconstruction = -2 * np.sum(residuals * dxdt_interp_i)
    # print("cog params:", cog_i.shape, cog_a.shape, np.size(cog_b))
    cog_pred = cog_i @ cog_a + cog_b
    cog_prior = lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)
    grad_cog = 2 * lambda_cog * np.sum(t_pred_i - cog_pred)

    loss = np.sum(residuals ** 2) + cog_prior
    grad = grad_reconstruction + grad_cog
    return loss, grad

def estimate_beta_for_patient(beta_i: float, X_obs_i: np.ndarray, dt_i: np.ndarray,
                              X_pred: np.ndarray, t_span: np.ndarray,
                              cog_i: np.ndarray, cog_a: np.ndarray, cog_b: float,
                              theta: np.ndarray, K: np.ndarray, lambda_cog: float = 0.0,
                              use_jacobian: bool = False, t_max: float = 12.0) -> float:
    """
    Estimates the optimal beta_i for a single patient.

    Parameters
    ----------
    # TODO: DOCSTRINGS
    Returns
    -------
    float
        Optimized beta_i value for the patient.
    """
    #beta_guess = np.median(dt_obs)
    beta_guess = beta_i
    
    if use_jacobian == True:
        loss_function = beta_loss_jac
        args=(X_obs_i, dt_i, X_pred, t_span, cog_i, cog_a, cog_b, theta, K, lambda_cog)
    else:
        loss_function = beta_loss
        args=(X_obs_i, dt_i, X_pred, t_span, cog_i, cog_a, cog_b, lambda_cog)

    result = minimize(
        loss_function,
        x0=beta_guess,
        args=args,
        jac=use_jacobian,
        bounds=[(0, t_max)],
        method="L-BFGS-B"
    )

    return result.x[0]
