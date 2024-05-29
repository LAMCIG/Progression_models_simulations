from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc
from ebm.likelihood import EventProbabilities
from ebm.transformer import ContinuousDistributionFitter
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import norm, uniform

class MCMCAnalysis:
    def __init__(self, X, y, distribution=norm, **dist_params):
        self.X = X
        self.y = y
        self.distribution = distribution
        self.dist_params = dist_params

    def perform_analysis(self):
        # Compute log probabilities
        log_p_e, log_p_not_e = log_distributions(self.X, self.y, point_proba=False, distribution=self.distribution, **self.dist_params)

        # Greedy ascent to find an initial good order
        starting_order = np.arange(self.X.shape[1])  # initialize order based on feature count
        ideal_order = np.arange(self.X.shape[1]) # I should try to use .copy()
        np.random.shuffle(starting_order)  # shuffle starting order

        order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e,
                                                     n_iter=10_000, order=starting_order,
                                                     random_state=2020)

        print(f"Greedy Ascent Result: {order}")

        try:
            orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e,
                                                         order=order, n_iter=500_000,
                                                         random_state=2020)

            if len(orders) == 0:
                raise ValueError("MCMC did not generate any orders.")
                
            first_ten_orders = orders[:10]
            rho, _ = spearmanr(ideal_order, orders[0])
            return first_ten_orders, rho, loglike, update_iters, probas

        except Exception as e:
            print(f"An error occurred during MCMC: {e}")
            return None, None, None, None, None

    def print_orders(self, orders, num_orders=10):
        print(f"First {num_orders} MCMC sampled orders:", orders[:num_orders])
