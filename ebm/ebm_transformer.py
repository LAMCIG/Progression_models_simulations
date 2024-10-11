from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc

class EBMModel(BaseEstimator):
    """
    Event-Based Model (EBM) for neurodegenerative disease progression estimation.
    
    Parameters:
        prior (np.ndarray): Prior knowledge about the disease progression order (default: None).
        random_state (int): Random seed for reproducibility.
        greedy_iters (int): Number of iterations for greedy ascent.
        mcmc_iters (int): Number of iterations for MCMC.
    
    Attributes:
        results (dict): Results of the EBM run, including orders and statistical correlations.
    """
    def __init__(self, prior=None, random_state=1, greedy_iters=10000, mcmc_iters=500000):
        self.prior = prior
        self.random_state = random_state
        self.greedy_iters = greedy_iters
        self.mcmc_iters = mcmc_iters
        self.results = None

    def fit(self, X, y=None):
        """
        Fit the EBM model using the input data and find the disease progression order.
        
        Parameters:
            X (np.ndarray): Feature matrix of biomarker values.
            y (np.ndarray): Disease stages or progression stages (optional, used for log distributions).
        """
        # Calculate log probabilities
        log_p_e, log_p_not_e = log_distributions(X, y, point_proba=False)
        
        rng = np.random.RandomState(self.random_state)
        ideal_order = np.arange(X.shape[1])
        starting_order = rng.choice(ideal_order, len(ideal_order), replace=False)
        starting_order_copy = starting_order.copy()

        # Greedy ascent
        order, loglike, _ = greedy_ascent(log_p_e, log_p_not_e, 
                                          n_iter=self.greedy_iters, 
                                          order=starting_order,
                                          prior=self.prior,
                                          random_state=self.random_state)
        greedy_order = order.copy()

        # MCMC
        orders, loglike, _, _ = mcmc(log_p_e, log_p_not_e,
                                     order=order, n_iter=self.mcmc_iters,
                                     prior=self.prior,
                                     random_state=self.random_state)
        
        if len(orders) != 0:
            best_order_idx = np.argmax(loglike)
            best_order = orders[best_order_idx].copy()
        else:
            print("Warning: MCMC did not accept new orders, returning greedy ascent result.")
            best_order = order.copy()

        # Calculate statistics
        starting_spearmanr, _ = spearmanr(ideal_order, starting_order_copy)
        greedy_spearmanr, _ = spearmanr(ideal_order, greedy_order)
        best_spearmanr, _ = spearmanr(ideal_order, best_order)

        starting_kendalltau, _ = kendalltau(ideal_order, starting_order_copy)
        greedy_kendalltau, _ = kendalltau(ideal_order, greedy_order)
        best_kendalltau, _ = kendalltau(ideal_order, best_order)

        # Store results
        self.results = {
            'starting_order': starting_order_copy,
            'greedy_order': greedy_order,
            'best_order': best_order,
            'starting_spearmanr': starting_spearmanr,
            'greedy_spearmanr': greedy_spearmanr,
            'best_spearmanr': best_spearmanr,        
            'starting_kendalltau': starting_kendalltau,
            'greedy_kendalltau': greedy_kendalltau,
            'best_kendalltau': best_kendalltau,
            'num_iters': len(orders)
        }

        return self

    def predict(self, X=None):
        """
        Predict the disease progression order based on the fitted EBM model.
        
        Returns:
            np.ndarray: The predicted best disease progression order.
        """
        if self.results is None:
            raise ValueError("The model has not been fitted yet.")
        
        return self.results['best_order']

    def score(self, X=None, y=None):
        """
        Score the model based on the Spearman correlation between the ideal and predicted orders.
        
        Returns:
            float: The Spearman correlation of the best predicted order.
        """
        if self.results is None:
            raise ValueError("The model has not been fitted yet.")
        
        return self.results['best_spearmanr']
