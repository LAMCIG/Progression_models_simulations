import numpy as np
from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc
from ebm.likelihood import EventProbabilities
from scipy.stats import spearmanr

class MCMCAnalysis:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def perform_analysis(self):
        log_p_e, log_p_not_e = log_distributions(self.X, self.y, point_proba=False)
        starting_order = np.arange(self.X.shape[1])
        np.random.shuffle(starting_order)

        order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e,
                                                     n_iter=10_000, order=starting_order,
                                                     random_state=2020)
       
        orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e,
                                                     order=order, n_iter=500_000,
                                                     random_state=2020)

        if not orders:
            print("Warning: No orders generated from MCMC.")
            return None

        first_ten_orders = orders[:10]
        rho, _ = spearmanr(order, orders[0] if orders else np.zeros_like(order))

        return first_ten_orders, rho, loglike, update_iters, probas
