import numpy as np
from .EBMAnalyzer import EBMAnalyzer
import scipy.stats as stats
class DiseaseProgressionAnalyzer:
    def __init__(self, patient_samples):
        self.patient_samples = patient_samples
    
    def run_analysis(self, analysis_type='ebm', **kwargs):
        if analysis_type == 'ebm':
            return self._run_ebm_analysis(**kwargs)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _run_ebm_analysis(self, distribution=stats.norm, **dist_params):
        # extract X and y from patient samples
        stages = np.array([sample[0] for sample in self.patient_samples])
        biomarkers = np.array([sample[1] for sample in self.patient_samples])
        
        # define health status based on stage: stages 1-3 are considered healthy, >3 are diseased
        n_healthy = sum(stages <= 3)
        y = np.array([0] * n_healthy + [1] * (len(stages) - n_healthy))
        
        # perform ebm analysis
        ebm_analyzer = EBMAnalyzer(biomarkers, y, distribution=distribution, **dist_params)
        orders, rho, loglike, update_iters, probas = ebm_analyzer.perform_analysis()
        
        if orders is not None:
            ebm_analyzer.print_orders(orders)
            rho_values = ebm_analyzer.compute_spearman_rho(orders)
            print(f"Spearman's rho values: {rho_values}")
        
        return orders, rho, loglike, update_iters, probas
    
    def print_orders(self, orders, num_orders=10):
        ebm_analyzer = EBMAnalyzer(self.patient_samples)
        ebm_analyzer.print_orders(orders, num_orders)

    def spearman_correlation(self, orders, true_order):
        ebm_analyzer = EBMAnalyzer(self.patient_samples)
        return ebm_analyzer.spearman_correlation(orders, true_order)

