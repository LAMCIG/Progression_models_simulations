class MCMCAnalysis:
    def __init__(self, simulation_instance):
        if not simulation_instance.X or not simulation_instance.y:
            raise ValueError("Simulation data not available. Ensure you have run simulate().")
        self.simulation = simulation_instance
        from ebm.probability import log_distributions
        self.log_p_e, self.log_p_not_e = log_distributions(self.simulation.X, self.simulation.y, point_proba=False)

    def run_greedy_ascent(self, start_order, n_iter=10000, random_state=2020):
        order, loglike, update_iters = greedy_ascent(self.data.log_p_e, self.data.log_p_not_e, 
                                                     n_iter=n_iter, order=start_order,
                                                     random_state=random_state, prior=self.priors)
        return order, loglike, update_iters

    def run_mcmc(self, start_order, n_iter=500000, random_state=2020):
        orders, loglike, update_iters, probas = mcmc(self.data.log_p_e, self.data.log_p_not_e,
                                                     order=start_order, n_iter=n_iter,
                                                     random_state=random_state, prior=self.priors)
        return orders, loglike, update_iters, probas

    ### statistics
    
    def compute_spearman_rho(orders, true_order):
        mean_rho = 0
        for order in orders:
            rho, _ = spearmanr(true_order, order)
            mean_rho += rho
        mean_rho /= len(orders)
        return mean_rho

    def analyze_rho_vs_lambda(orders, true_order, lambda_range):
        spearman_rhos = []
        for lambda_value in lambda_range:
            scaled_log_prior = compute_log_prior_from_adjacency(orders.A, lambda_value)
            rho, _ = spearmanr(true_order, orders[0])  # Example using first order
            spearman_rhos.append(rho)
        return spearman_rhos
    
    ### plotting
    
    def plot_spearman_rho_vs_lambda(lambda_values, spearman_rhos):
        plt.figure(figsize=(10, 6))
        plt.semilogx(lambda_values, spearman_rhos, marker='o', linestyle='-', color='blue')
        plt.xlabel('Lambda (λ)')
        plt.ylabel("Spearman's Rho")
        plt.title("Spearman's Rho vs. Lambda (λ) for MCMC-derived Biomarker Orders")
        plt.grid(True)
        plt.show()

