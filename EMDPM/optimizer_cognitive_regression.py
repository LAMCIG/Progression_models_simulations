import numpy as np
import pandas as pd
from scipy.optimize import minimize

# TODO: type hints
# TODO: doc strings

def fit_linear_cog_regression_multi(cog: np.ndarray, dt: np.ndarray, 
                                    beta: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, float]:
    #assert cog.shape[0] == dt.shape[0]
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt + beta[index_array]
    
    if cog.shape[0] == 1:
        cog = cog.T
        
    X = np.hstack([cog, np.ones((cog.shape[0], 1))])  # (n_obs, n_features + 1)
    #print(X.T.shape, t_ij.shape)
    
    XtX = X.T @ X
    XtY = X.T @ t_pred
    new_cog = np.linalg.inv(XtX) @ XtY

    a = new_cog[:-1]
    b = new_cog[-1]

    return a, b
