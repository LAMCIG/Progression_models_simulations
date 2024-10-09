from model_generator.base_disease_model import BaseDiseaseModel
import numpy as np

class SigmoidModel(BaseDiseaseModel):
    def fit(self, X=None, y=None):
        n_stages = self.params.get('n_stages', 10)
        biomarker_params = self.params.get('biomarker_params', {})
        
        self.model_values = np.zeros((n_stages, 100))
        for stage, params in biomarker_params.items():
            s, c = params['s'], params['c']
            self.model_values[stage] = 1 / (1 + np.exp(-s * (np.arange(100) - c)))
        
        return self