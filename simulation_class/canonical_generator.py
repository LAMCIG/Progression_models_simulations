import numpy as np
import matplotlib.pyplot as plt
from biomarker_utils import generate_transition_matrix, apply_transition_matrix2, solve_ode_system

class CanonicalGenerator:
    def __init__(self, n_biomarkers, n_stages, model_type='sigmoid', biomarkers_params=None):
        self.n_biomarkers = n_biomarkers
        self.n_stages = n_stages
        self.model_type = model_type
        self.biomarkers_params = biomarkers_params
        self.model = self._generate_model()
    
    def _generate_model(self):
        if self.model_type == 'sigmoid':
            return self._generate_sigmoid_model()
        elif self.model_type == 'transition_matrix':
            return self._generate_transition_matrix_model()
        elif self.model_type == 'ode':
            return self._generate_ode_model()
        else:
            raise ValueError(f"Model does not exist: {self.model_type}")
    
    def _generate_sigmoid_model(self):
        def sigmoid(stage, s, c):
            return 1 - 1 / (1 + np.exp(-(stage-s)/c))
        
        model = {}
        for biomarker, params in self.biomarkers_params.items():
            s = params['s']
            c = params['c']
            model[biomarker] = [sigmoid(stage, s, c) for stage in range(self.n_stages)]
        return model
    
    def _generate_transition_matrix_model(self):
        model = {}
        transition_matrix = generate_transition_matrix(self.n_biomarkers, self.biomarkers_params.get('coeff', 1e-1))
        for biomarker in range(self.n_biomarkers):
            model[biomarker] = apply_transition_matrix2(transition_matrix, range(self.n_stages), np.ones(self.n_biomarkers))
        return model
    
    def _generate_ode_model(self):
        def ode_model(stage, rate):
            return np.exp(-rate * stage)
        
        model = {}
        for biomarker in range(self.n_biomarkers):
            rate = np.random.rand()
            model[biomarker] = [ode_model(stage, rate) for stage in range(self.n_stages)]
        return model
    
    def model_predict(self, stage, biomarker):
        return self.model[biomarker][stage]
    
    def plot_disease_progression(self):
        plt.figure()
        for biomarker in range(self.n_biomarkers):
            plt.plot(range(self.n_stages), self.model[biomarker], label=f'Biomarker {biomarker + 1}')
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.title('Disease Progression of Biomarkers')
        plt.legend()
        plt.show()
