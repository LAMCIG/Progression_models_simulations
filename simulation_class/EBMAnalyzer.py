import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm, spearmanr, kendalltau
from .ebm.probability_legacy import log_distributions, predict_stage
from .ebm.mcmc import greedy_ascent, mcmc
from .ebm.likelihood import EventProbabilities
import matplotlib.pyplot as plt
class EBMAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, distribution="norm", prior = None, **dist_params):
        self.distribution = distribution
        self.dist_params = dist_params
        self.prior = prior
        
        ## mcmc results
        self.orders = None
        self.loglike = None
        self.update_iters = None
        self.probas = None
        self.log_p_e = None
        self.log_p_not_e = None
        
        ## evaluation metrics
        self.spearman_rho = None
        self.kendall_tau = None

    def fit(self, X, y=None):
        
        # validate that X and y contain only finite values
        if not np.all(np.isfinite(X)):
            raise ValueError("The data in X contains non-finite values (NaN or Inf).")
        if y is not None and not np.all(np.isfinite(y)):
            raise ValueError("The data in y contains non-finite values (NaN or Inf).")
        
        self.log_p_e, self.log_p_not_e, \
        cdf_p_e, cdf_p_not_e, left_min, right_max, flip_vec = log_distributions(X, y, 
                                                 point_proba=False, 
                                                 distribution=self.distribution, 
                                                 **self.dist_params)
        self.fitted_cdfs = []
        self.fitted_cdfs.append(cdf_p_e)
        self.fitted_cdfs.append(cdf_p_not_e)
        self.fitted_cdfs.append(left_min)
        self.fitted_cdfs.append(right_max)
        self.fitted_cdfs.append(flip_vec)

        starting_order = np.arange(X.shape[1])
        ideal_order = np.arange(X.shape[1])
        np.random.shuffle(starting_order)
        starting_tau, _ = kendalltau(ideal_order, starting_order)
        print(f"Starting Order: {starting_order}, kendall-tau:{starting_tau}")

        order, loglike, update_iters = greedy_ascent(self.log_p_e, 
                                                     self.log_p_not_e, 
                                                     n_iter=10_000, 
                                                     order=starting_order,
                                                     prior = self.prior, 
                                                     random_state=2020
                                                     )
        
        print(f"Greedy Ascent Result: {starting_order}")

        orders, loglike, update_iters, probas = mcmc(self.log_p_e, 
                                                     self.log_p_not_e, 
                                                     order=order, 
                                                     n_iter=500_000,
                                                     prior = self.prior,
                                                     random_state=2020
                                                     )
        
        
        if len(loglike) > 0:
            best_order_idx = np.argmax(loglike)
            self.orders = orders
        else: # contingency for when no orders are generated
            best_order_idx = 0
            self.orders = starting_order
        
        
        print(ideal_order)
        best_order = np.array(self.orders[best_order_idx]).flatten()
        print(best_order)
        self.spearman_rho, _ = spearmanr(np.array(ideal_order, dtype=np.float64), np.array(best_order, dtype=np.float64))
        self.kendall_tau, _ = kendalltau(ideal_order, best_order)
        self.loglike = loglike
        self.update_iters = update_iters
        self.probas = probas

        return self

    def transform(self, X, y=None):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        self.best_order = self.orders[np.argmax(self.loglike)]

        log_p_e, log_p_not_e, cdf_p_e, cdf_p_not_e, \
        left_min, right_max, flip_vec = log_distributions(X, y, 
                                                 point_proba=False, 
                                                 distribution=self.distribution, 
                                                 fitted_cdfs = self.fitted_cdfs,
                                                 **self.dist_params)

        likelihood_matrix = predict_stage(self.best_order, log_p_e, log_p_not_e)
        return likelihood_matrix
    
    def get_params(self):
        return {
            'orders': self.orders,
            'loglike': self.loglike,
            'update_iters': self.update_iters,
            'probas': self.probas,
            'best_order': self.best_order,
            'spearman_rho': self.spearman_rho,
            'kendall_tau': self.kendall_tau
        }
        
    def summary(self):
        if self.orders is None or self.best_order is None:
            print("No results to summarize. Make sure to run fit() successfully.")
            return
        print(f"Best Order: {self.best_order}")
        print(f"Spearman's Rho: {self.spearman_rho}")
        print(f"Kendall's Tau: {self.kendall_tau}")

    def print_orders(self, num_orders=10):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        print(f"First {num_orders} MCMC sampled orders:", self.orders[:num_orders])
        
    def get_orders(self):
        return self.orders
    
    def get_loglike(self):
        return self.loglike

        

