import numpy as np
import pandas as pd
from scipy.optimize import minimize

# TODO: type hints
# TODO: doc strings

def fit_linear_cog_regression_multi(cog: np.ndarray, dt: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, float]:
    
    #assert cog.shape[0] == dt.shape[0]
    t_ij = dt + beta  # (n_obs,)

    # intercept column
    X = np.hstack([cog, np.ones((cog.shape[0], 1))])  # (n_obs, n_features + 1)

    XtX = X.T @ X
    XtY = X.T @ t_ij
    theta = np.linalg.pinv(XtX) @ XtY

    a = theta[:-1]
    b = theta[-1]

    return a, b
