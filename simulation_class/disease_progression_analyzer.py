from scipy.stats import spearmanr

class DiseaseProgressionAnalyzer:
    def __init__(self, patient_samples):
        self.patient_samples = patient_samples
    
    def run_analysis(self, analysis_type='mcmc'):
        if analysis_type == 'mcmc':
            return self._run_mcmc_analysis()
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _run_mcmc_analysis(self):
        # TODO: reintegrate mcmc
        pass
    
    def spearman_correlation(self, order, ideal_order):
        """computes spearman's rank correlation between the provided order and the ideal order."""
        return spearmanr(order, ideal_order)

    def print_orders(self, orders, num_orders=10):
        """prints the first num_orders orders."""
        print(f"First {num_orders} MCMC sampled orders:", orders[:num_orders])
