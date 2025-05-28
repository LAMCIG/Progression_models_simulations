import numpy as np
import pandas as pd
from scipy.optimize import minimize

# TODO: type hints
# TODO: doc strings

def cog_loss(params, t_ij, s_ij, x_obs, x_pred, lambda_cog):
    """
    Compute total loss for (a, b): data reconstruction + cognitive regression mismatch.
    """
    a, b = params
    residuals_x = x_obs - x_pred
    residuals_t = t_ij - (a * s_ij + b)
    return np.sum(residuals_x**2) + lambda_cog * np.sum(residuals_t**2)

def cog_loss_jac(params, t_ij, s_ij, x_obs, x_pred, lambda_cog):
    """
    Compute loss and gradient for cognitive regression.
    """
    a, b = params
    residuals_x = x_obs - x_pred
    residuals_t = t_ij - (a * s_ij + b)

    grad_a = -2 * lambda_cog * np.sum(residuals_t * s_ij)
    grad_b = -2 * lambda_cog * np.sum(residuals_t)

    loss = np.sum(residuals_x**2) + lambda_cog * np.sum(residuals_t**2)
    grad = np.array([grad_a, grad_b])
    return loss, grad

def fit_optimizer_regression(df, beta_iter, iteration, x_reconstructed, t_span,
                             lambda_cog, s_fit, use_jacobian=False, a_guess = 1, b_guess = 0):
    """
    Optimize global cognitive regression parameters (a, b) given β_i and model predictions.
    """
    df = df.copy()
    df["beta"] = beta_iter[str(iteration)]
    df["t_ij"] = df["dt"] + df["beta"]

    t_ij = df["t_ij"].values
    s_ij = df["cognitive_score"].values
    x_obs = df[[col for col in df.columns if "biomarker_" in col]].values

    n_biomarkers = x_reconstructed.shape[0]
    x_pred = np.zeros_like(x_obs)

    for j in range(n_biomarkers):
        interpolated = np.interp(t_ij, t_span, x_reconstructed[j])
        x_pred[:, j] = s_fit[j] * interpolated

    result = minimize(
        cog_loss_jac if use_jacobian else cog_loss,
        x0=np.array([a_guess, b_guess]),
        args=(t_ij, s_ij, x_obs, x_pred, lambda_cog),
        method="L-BFGS-B",
        bounds=([0.01,6],[0.01,12]), 
        jac=use_jacobian
    )

    return tuple(result.x)
