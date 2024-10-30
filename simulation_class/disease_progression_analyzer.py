import numpy as np
import pandas as pd
from .EBMAnalyzer import EBMAnalyzer
import scipy.stats as stats
class DiseaseProgressionAnalyzer:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initializes the DiseaseProgressionAnalyzer with biomarker data (X) and labels (y).
        
        Parameters:
        X (np.ndarray): Biomarker data for each patient.
        y (np.ndarray): Labels indicating health.
        """
        self.X = X
        self.y = y
        self.ebm_analyzer = None
        self.prior = None

    def run_analysis(self, analysis_type: str):
        """
        Runs specified analysis on the patient data.
        """
        if analysis_type == 'ebm':
            self.ebm_analyzer = EBMAnalyzer(prior=self.prior)
            self.ebm_analyzer.fit(self.X, self.y) # im not a fan of the variable name but it will do
            likelihood_matrix = self.ebm_analyzer.transform(self.X, self.y)
            return likelihood_matrix
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def set_prior(self, prior):
        self.prior = prior
        
    def print_orders(self, num_orders=10):
        if self.ebm_analyzer is None:
            raise ValueError("EBMAnalyzer not fitted. Run run_analysis() first.")
        self.ebm_analyzer.print_orders(num_orders)
