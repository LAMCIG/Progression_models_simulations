import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm, spearmanr
from .ebm.probability import log_distributions, predict_stage
from .ebm.mcmc import greedy_ascent, mcmc
from .ebm.likelihood import EventProbabilities

class EBMAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, distribution=norm, **dist_params):
        self.distribution = distribution
        self.dist_params = dist_params
        self.orders = None
        self.rho = None
        self.loglike = None
        self.update_iters = None
        self.probas = None
        self.log_p_e = None
        self.log_p_not_e = None

    def fit(self, X, y=None):
        self.log_p_e, self.log_p_not_e = log_distributions(X, y, 
                                                 point_proba=False, 
                                                 distribution=self.distribution, 
                                                 **self.dist_params)

        starting_order = np.arange(X.shape[1])
        np.random.shuffle(starting_order)

        order, loglike, update_iters = greedy_ascent(self.log_p_e, 
                                                     self.log_p_not_e, 
                                                     n_iter=10_000, 
                                                     order=starting_order, 
                                                     random_state=2020)
        
        print(f"Greedy Ascent Result: {order}")

        orders, loglike, update_iters, probas = mcmc(self.log_p_e, 
                                                     self.log_p_not_e, 
                                                     order=order, 
                                                     n_iter=500_000, 
                                                     random_state=2020)

        self.orders = orders if len(orders) > 0 else [starting_order]
        ideal_order = np.arange(X.shape[1])
        self.rho, _ = spearmanr(ideal_order, self.orders[np.argmax(loglike)])
        self.loglike = loglike
        self.update_iters = update_iters
        self.probas = probas

        return self

    def transform(self, X, y=None):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        likelihood_matrix = predict_stage(self.orders[np.argmax(self.loglike)], self.log_p_e, self.log_p_not_e)
        return likelihood_matrix, self.orders, self.rho, self.loglike, self.update_iters, self.probas
    
    def get_params(self):
        return {
            'orders': self.orders,
            'loglike': self.loglike,
            'update_iters': self.update_iters,
            'probas': self.probas
        }

    def print_orders(self, num_orders=10):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        print(f"First {num_orders} MCMC sampled orders:", self.orders[:num_orders])

    def spearman_correlation(self):
        return self.rho
