import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm, spearmanr, kendalltau
from .ebm.probability import log_distributions, predict_stage
from .ebm.mcmc import greedy_ascent, mcmc
from .ebm.likelihood import EventProbabilities

class EBMAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, distribution=norm, prior = None, **dist_params):
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
                                                     prior = self.prior, 
                                                     random_state=2020
                                                     )
        
        print(f"Greedy Ascent Result: {order}")

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
        else:
            best_order_idx = 0
            self.orders = [order]
        
        ideal_order = np.arange(X.shape[1])
        self.rho, _ = spearmanr(ideal_order, self.orders[best_order_idx])
        self.kendall_tau, _ = kendalltau(ideal_order, self.orders[best_order_idx])
        self.loglike = loglike
        self.update_iters = update_iters
        self.probas = probas

        return self

    def transform(self, X, y=None):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        self.best_order = self.orders[np.argmax(self.loglike)]
        likelihood_matrix = predict_stage(self.best_order, self.log_p_e, self.log_p_not_e)
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
        print(f"Best Order: {self.best_order}")
        print(f"Spearman's Rho: {self.spearman_rho}")
        print(f"Kendall's Tau: {self.kendall_tau}")

    def print_orders(self, num_orders=10):
        if self.orders is None:
            raise ValueError("No orders found. Run fit() first.")
        print(f"First {num_orders} MCMC sampled orders:", self.orders[:num_orders])
