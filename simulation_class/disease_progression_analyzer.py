import numpy as np
from .EBMAnalyzer import EBMAnalyzer
import scipy.stats as stats

class DiseaseProgressionAnalyzer:
    def __init__(self, patient_samples):
        self.patient_samples = patient_samples
        self.X, self.y = self._prepare_data(patient_samples)
        self.ebm_analyzer = None

    def _prepare_data(self, patient_samples):
        stages = np.array([sample[0] for sample in patient_samples])
        biomarkers = np.array([sample[1] for sample in patient_samples])
        n_healthy = sum(stages <= 3)  # adjust threshold as needed
        y = np.array([0] * n_healthy + [1] * (len(stages) - n_healthy))
        return biomarkers, y

    def run_analysis(self, analysis_type='ebm'):
        if analysis_type == 'ebm':
            self.ebm_analyzer = EBMAnalyzer()
            self.ebm_analyzer.fit(self.X, self.y)
             # im not a fan of the variable name but it will do
            likelihood_matrix = self.ebm_analyzer.transform(self.X)
            return likelihood_matrix
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def print_orders(self, num_orders=10):
        if self.ebm_analyzer is None:
            raise ValueError("EBMAnalyzer not fitted. Run run_analysis() first.")
        self.ebm_analyzer.print_orders(num_orders)
