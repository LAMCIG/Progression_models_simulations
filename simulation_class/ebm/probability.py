import numpy as np


def fit_distributions(X, y, normalize=False):
    """Fit distribution p(x|E), p(x|~E) as a mixture of Gaussian and Uniform, see Fonteijn 
    section `Mixture models for the data likelihood`. 
    - P(x|E) = P(x > X | E)
    - P(x|~E) = P(x < X| ~E)
    """
    # TODO: not sure about how to compute probabilities
    from scipy.stats import norm, uniform
    if normalize:
        X = X / X.max(axis=1)[:, np.newaxis]
    
    avg = X[y==0, ...].mean(axis=0)
    std = X[y==0, ...].std(axis=0)
    p_not_e = [norm(loc, s) for loc, s in zip(avg, std)]

    left_min = X.min(axis=0)
    right_max = avg.copy()
    p_e = [uniform(m1, m2-m1) for m1, m2 in zip(left_min, right_max)]
    return p_e, p_not_e, left_min, right_max


def log_distributions(X, y, point_proba=False, *, X_test=None, y_test=None, normalize=False, eps=1e-8):
    """Precomute probabilities for all features."""
    X = np.array(X).astype(np.float64)
    y = np.array(y)
    cdf_p_e, cdf_p_not_e, left_min, right_max = fit_distributions(X, y, normalize=normalize)
    
    if X_test is not None:
        X = np.array(X_test).astype(np.float64)
        y = np.array(y_test)
        
    n, m = X.shape
    log_p_e, log_p_not_e = np.zeros_like(X), np.zeros_like(X)
    
    
    if not point_proba:
        for i in range(n):
            for j in range(m):
                p = cdf_p_e[j].cdf(np.clip(X[i, j], left_min[j]+eps, right_max[j]-eps))
                log_p_e[i,j] = np.log(np.clip(1 - p, eps, 1-eps))
                
                p = cdf_p_not_e[j].cdf(X[i, j])
                log_p_not_e[i,j] = np.log(np.clip(p, eps, 1-eps))
    else:
        for i in range(n):
            for j in range(m):
                window = np.abs(right_max[j] - left_min[j]) / 20
                
                p_left = cdf_p_e[j].cdf(np.clip(X[i, j], left_min[j]+eps, right_max[j]-eps)-window/2)
                p_right = cdf_p_e[j].cdf(np.clip(X[i, j], left_min[j]+eps, right_max[j]-eps)+window/2)
                log_p_e[i,j] = np.log(np.clip(p_right - p_left, eps, 1-eps))
                
                p_left = cdf_p_not_e[j].cdf(X[i, j]-window/2)
                p_right = cdf_p_not_e[j].cdf(X[i, j]+window/2)
                log_p_not_e[i,j] = np.log(np.clip(p_right - p_left, eps, 1-eps))
    return log_p_e, log_p_not_e


def predict_stage(event_order, log_p_e, log_p_not_e):
    likelihood = []
    for k in range(len(event_order)):
        likelihood.append(log_p_e[:, event_order[k]]- log_p_not_e[:, event_order[k]]) 
    return np.array(likelihood)
