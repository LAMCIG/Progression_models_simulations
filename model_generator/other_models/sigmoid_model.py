from model_generator.base_disease_model import BaseDiseaseModel
import numpy as np
class SigmoidModel(BaseDiseaseModel):
    def __init__(self, **params):
        super().__init__(**params)
    
    def fit(self, X=None, y=None):
        n_stages = self.params.get('n_stages', 10)
        biomarker_params = self.params.get('biomarker_params', {})
        
        time_points = np.linspace(self.start_time, self.end_time, self.steps)
        self.model_values = np.zeros((n_stages, len(time_points)))
        
        for stage, params in biomarker_params.items():
            s, c = params['s'], params['c']
            self.model_values[stage] = 1 / (1 + np.exp(-s * (time_points - c)))

        adjusted_time_points = self._dynamic_time_adjustment(self.model_values)
        
        if len(adjusted_time_points) > len(time_points):
            self.model_values = np.zeros((n_stages, len(adjusted_time_points)))
            for stage, params in biomarker_params.items():
                s, c = params['s'], params['c']
                self.model_values[stage] = 1 / (1 + np.exp(-s * (adjusted_time_points - c)))
        
        return self