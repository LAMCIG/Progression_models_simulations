import numpy as np
import matplotlib.pyplot as plt
from .biomarker_utils import generate_transition_matrix, initialize_biomarkers, simulate_progression_over_stages
from .ode_generator import ODEGenerator

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
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_sigmoid_model(self):
        def sigmoid(x, s, c):
            return 1 / (1 + np.exp(-s * (x - c)))

        model = {}
        for biomarker, params in self.biomarkers_params.items():
            s = params['s']
            c = params['c']
            model[biomarker] = [sigmoid(stage, s, c) for stage in range(self.n_stages)]
        return model
    
    def _generate_transition_matrix_model(self):
        coeff = self.biomarkers_params.get('coeff', 1e-1)
        transition_matrix = generate_transition_matrix(self.n_biomarkers, coeff)
        y_init = initialize_biomarkers(self.n_biomarkers)
        stages = np.arange(self.n_stages)
        model = simulate_progression_over_stages(transition_matrix, stages, y_init)
        return model
    
    def _generate_ode_model(self):
        ode_generator = ODEGenerator(self.n_biomarkers)
        t, x = ode_generator.multi_logistic_sym_force()
        self.time_points = t
        model = {i: x[i] for i in range(self.n_biomarkers)}
        return model
    
    def model_predict(self, stage, biomarker):
        if self.model_type == 'sigmoid':
            return self.model[biomarker][stage]
        elif self.model_type == 'transition_matrix':
            return self.model[biomarker][stage]
        elif self.model_type == 'ode':
            idx = np.searchsorted(self.time_points, stage)
            return self.model[biomarker][idx]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def plot_disease_progression(self):
        plt.figure()
        for biomarker in range(self.n_biomarkers):
            if self.model_type == 'ode':
                plt.plot(self.time_points, self.model[biomarker], label=f'Biomarker {biomarker}')
            else:
                plt.plot(range(self.n_stages), self.model[biomarker], label=f'Biomarker {biomarker}')
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.legend()
        plt.show()
