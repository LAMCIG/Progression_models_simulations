import numpy as np
from .transformer import ContinuousDistributionFitter
from scipy.stats import norm, uniform, expon, lognorm # <-- add more distributions here

# TODO: docstrings
def fit_distribution(data, dist_name = "norm"):
    dist_dict = {
        "norm": norm,
        "uniform": uniform,
        "expon": expon,
        "lognorm": lognorm,
      # "'example string': example scipy distribution"
      
      # You may add which ever distribution you would like to use here,
      # DO NOT FORGET TO IMPORT YOUR DISTRIBUTION FROM SCIPY.STATS  
    }
    
    if dist_name not in dist_dict:
        raise ValueError(f"Unsupported distribution type: {dist_name}\nPlease, make sure your desired distribution has been imported.")
    dist = dist_dict[dist_name]
    params = dist.fit(data)
    return dist, params
    
    

def fit_distributions(X, y, normalize=False, distribution=norm, **dist_params):
    """Fit distribution p(x|E), p(x|~E) as a mixture of Gaussian and Uniform, see Fonteijn 
    section `Mixture models for the data likelihood`. 
    - P(x|E) = P(x > X | E)
    - P(x|~E) = P(x < X| ~E)
    """
    if normalize:
        X = X / X.max(axis=1)[:, np.newaxis]

    fitter_not_e = ContinuousDistributionFitter(distribution, **dist_params)
    X_not_e = X[y == 0]
    fitter_not_e.fit(X_not_e)

    fitter_e = ContinuousDistributionFitter(uniform)
    X_e = X[y == 1]
    fitter_e.fit(X_e)

    return fitter_e, fitter_not_e

# https://stackoverflow.com/questions/5365520/numpy-converting-array-from-float-to-strings
# https://stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us
# https://medium.com/anu-perumalsamy/log-likelihood-function-by-hand-r-and-python-with-cheat-sheet-3541effbcd88

def log_distributions(X, y, point_proba=False, *, X_test=None, y_test=None, normalize=False, eps=1e-8, distribution=norm, **dist_params):
    """precompute probabilities for all features."""
    
    # convert input data to float64 numpy arrays
    X = np.array(X).astype(np.float64)
    y = np.array(y)
    
    # fit distributions to the data
    fitter_e, fitter_not_e = fit_distributions(X, y, normalize=normalize, distribution=distribution, **dist_params)

    # if test data is provided, use it
    if X_test is not None:
        X = np.array(X_test).astype(np.float64)
        y = np.array(y_test)

    n, m = X.shape  # number of samples and number of features
    log_p_e, log_p_not_e = np.zeros_like(X), np.zeros_like(X)  # initialize log probability matrices

    # determine if we need to flip the cdf by comparing to median 
    flip = np.median(fitter_not_e.transform(X), axis=0) > np.median(fitter_e.transform(X), axis=0)

    if not point_proba:
        # calculate log_probabilities w/ out using point probabilities
        for i in range(n):  # iterate over samples
            for j in range(m):  # iterate over features
                # calculate the probability for the "diseased" distribution
                p = fitter_e.cdf(np.clip(X[i, j], fitter_e.lower_bound[j] + eps, fitter_e.upper_bound[j] - eps), j)
                if flip[j]:  # if flip is true, invert the probability
                    p = 1 - p
                log_p_e[i, j] = np.log(np.clip(1 - p, eps, 1 - eps))  # store the log probability

                # calculate the probability for the "healthy" distribution
                p = fitter_not_e.cdf(X[i, j], j)
                if flip[j]:  # if flip is true, invert the probability
                    p = 1 - p
                log_p_not_e[i, j] = np.log(np.clip(p, eps, 1 - eps))  # store the log probability
    else:
        # calculate log_probabilities w/ point probabilities
        for i in range(n):  # samples
            for j in range(m):  # features
                window = np.abs(fitter_e.upper_bound[j] - fitter_e.lower_bound[j]) / 20  # define a window for point probability

                # calculate the log probability for the "diseased" distribution
                p_left = fitter_e.cdf(np.clip(X[i, j], fitter_e.lower_bound[j] + eps, fitter_e.upper_bound[j] - eps) - window / 2, j)
                p_right = fitter_e.cdf(np.clip(X[i, j], fitter_e.lower_bound[j] + eps, fitter_e.upper_bound[j] - eps) + window / 2, j)
                log_p_e[i, j] = np.log(np.clip(p_right - p_left, eps, 1 - eps))

                # calculate the log probability for the "healthy" distribution
                p_left = fitter_not_e.cdf(X[i, j] - window / 2, j)
                p_right = fitter_not_e.cdf(X[i, j] + window / 2, j)
                log_p_not_e[i, j] = np.log(np.clip(p_right - p_left, eps, 1 - eps))
    
    return log_p_e, log_p_not_e

def predict_stage(event_order, log_p_e, log_p_not_e): # from anvars old code
    likelihood = []
    for k in range(len(event_order)):
        likelihood.append(log_p_e[:, event_order[k]]- log_p_not_e[:, event_order[k]]) 
    likelihood = np.array(likelihood)
    
    # conversion from log likelihood --> probablility
    likelihood = np.exp(likelihood)
    likelihood /= likelihood.sum(axis=0, keepdims=True)
    
    return likelihood