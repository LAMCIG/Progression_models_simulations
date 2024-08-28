import numpy as np
from scipy.stats import norm, uniform, expon, lognorm # <-- add more distributions here

# Dictionary of supported distributions
dist_dict = {
    "norm": norm,
    "uniform": uniform,
    "expon": expon,
    "lognorm": lognorm,
    # Add more distributions here if needed
}

# TODO: docstrings
# TODO: Return to how fonteijn does it but with allow for change in direction

def fit_distribution(data, dist_name="lognorm"): # TODO? should I pass a scipy distribution object
    """
    Fit a specified distribution to the data.
    
    Parameters:
    data (np.ndarray): The data to fit.
    dist_name (str): The name of the distribution to fit.
    
    Returns:
    dist (scipy.stats distribution): The fitted distribution.
    params (tuple): The parameters of the fitted distribution.
    """
    if dist_name not in dist_dict:
        raise ValueError(f"Unsupported distribution type: {dist_name}\nPlease ensure your desired distribution has been imported.")
    
    dist = dist_dict[dist_name]
    params = dist.fit(data)
    return dist, params

# https://stackoverflow.com/questions/5365520/numpy-converting-array-from-float-to-strings
# https://stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us
# https://medium.com/anu-perumalsamy/log-likelihood-function-by-hand-r-and-python-with-cheat-sheet-3541effbcd88

def log_distributions(X, y, point_proba=False, distribution="norm", normalize=False, eps=1e-8):
    """
    Precompute log probabilities for all features using specified distribution.

    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Labels (0 or 1).
    point_proba (bool): Whether to use point probabilities.
    dist_name (str): The name of the distribution to use.
    normalize (bool): Whether to normalize X.
    eps (float): Small value to avoid log(0).

    Returns:
    log_p_e (np.ndarray): Log probabilities for event (diseased) distribution.
    log_p_not_e (np.ndarray): Log probabilities for non-event (healthy) distribution.
    """
    if normalize:
        X = X / X.max(axis=1)[:, np.newaxis]

    # Fit distributions for event (diseased) and non-event (healthy)
    event_data = X[y == 1]
    non_event_data = X[y == 0]

    event_dist, event_params = fit_distribution(event_data.flatten(), distribution) # TODO: ditch flatten()
    non_event_dist, non_event_params = fit_distribution(non_event_data.flatten(), distribution)

    n, m = X.shape
    log_p_e, log_p_not_e = np.zeros_like(X), np.zeros_like(X)

    flip = np.median(event_dist.cdf(X), axis=0) > np.median(non_event_dist.cdf(X), axis=0)

    for i in range(n):
        for j in range(m):
            # Calculate log probabilities
            if not point_proba:
                p = event_dist.cdf(X[i, j])
                if flip[j]:
                    p = 1 - p
                log_p_e[i, j] = np.log(np.clip(1 - p, eps, 1 - eps))

                p = non_event_dist.cdf(X[i, j])
                if flip[j]:
                    p = 1 - p
                log_p_not_e[i, j] = np.log(np.clip(p, eps, 1 - eps))
            else:
                window = np.abs(event_dist.ppf(0.95) - event_dist.ppf(0.05)) / 20
                p_left = event_dist.cdf(X[i, j] - window / 2)
                p_right = event_dist.cdf(X[i, j] + window / 2)
                log_p_e[i, j] = np.log(np.clip(p_right - p_left, eps, 1 - eps))

                p_left = non_event_dist.cdf(X[i, j] - window / 2)
                p_right = non_event_dist.cdf(X[i, j] + window / 2)
                log_p_not_e[i, j] = np.log(np.clip(p_right - p_left, eps, 1 - eps))
    
    return log_p_e, log_p_not_e

def predict_stage(event_order, log_p_e, log_p_not_e):
    """
    Predict the stage of the event based on the log probabilities.

    Parameters:
    event_order (list): The order of events.
    log_p_e (np.ndarray): Log probabilities for the event.
    log_p_not_e (np.ndarray): Log probabilities for non-event.

    Returns:
    likelihood (np.ndarray): Predicted likelihood for each stage.
    """
    likelihood = []
    for k in range(len(event_order)):
        likelihood.append(log_p_e[:, event_order[k]] - log_p_not_e[:, event_order[k]])
    likelihood = np.array(likelihood)
    
    # Convert log likelihood to probability
    likelihood = np.exp(likelihood)
    likelihood /= likelihood.sum(axis=0, keepdims=True)
    
    return likelihood