import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple 
from .biomarker_utils import generate_transition_matrix, initialize_biomarkers, simulate_progression_over_stages
from .ode_generator import ODEGenerator

class CanonicalGenerator:
    def __init__(self, n_biomarker_stages: int, model_type: str = 'sigmoid', biomarkers_params: Dict = None):
        """
        Initializes the CanonicalGenerator with the specified parameters.
        """
        # User-defined member variables
        self.n_biomarker_stages = n_biomarker_stages
        self.model_type = model_type
        self.biomarkers_params = biomarkers_params
        
        # Output variables
        self.model_values = self._generate_model()
        self.biomarker_values = self._get_discrete_stage_values()

    def _generate_model(self) -> np.array:
        if self.model_type == 'sigmoid':
            return self._generate_sigmoid_model()
        elif self.model_type == 'transition_matrix':
            return self._generate_transition_matrix_model()
        elif self.model_type == 'ode':
            return self._generate_ode_model()
        # elif self.model_type == 'example':
        #     return self._generate_example_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_sigmoid_model(self) -> np.ndarray:
        def sigmoid(x, s, c):
            return 1 / (1 + np.exp(-s * (x - c)))

        x = np.linspace(-50, 120, 1000) # arbitrary domain for sigmoid model
        model_values = np.zeros((self.n_biomarker_stages, len(x)))
        
        for biomarker, params in self.biomarkers_params.items():
            s = params['s']
            c = params['c']
            model_values[biomarker] = sigmoid(x, s, c)

        return model_values
    
    def _generate_transition_matrix_model(self) -> np.ndarray:
        """
        Generates a transition matrix model for the biomarkers.

        Returns:
        --------
        np.ndarray
            The generated transition matrix model.
        """
        coeff = self.biomarkers_params.get('coeff')
        timespan = np.arange(40)
        n_biomarkers = self.n_biomarker_stages + 1 # generate extra to remove later
        
        transition_matrix = generate_transition_matrix(n_biomarkers, coeff)
        y_init = initialize_biomarkers(n_biomarkers)
        model_values = simulate_progression_over_stages(transition_matrix, timespan, y_init)
        return model_values.T[1:]
    
    def _generate_ode_model(self) -> np.ndarray:
        ode_generator = ODEGenerator(self.n_biomarker_stages)
        t, x = ode_generator.multi_logistic_sym_force()
        self.time_points = t
        return x # x is just the model_value at each timepoint t
    
    def _get_discrete_stage_values(self) -> np.ndarray:
        if self.model_type == 'ode':
            return np.array([np.interp(np.arange(self.n_biomarker_stages), self.time_points, self.model_values[i]) for i in range(len(self.model_values))])
        else:
            return self.model_values[:, np.linspace(0, self.model_values.shape[1] - 1, self.n_biomarker_stages, dtype=int)]
    
    def model_predict(self, stage: int, biomarker: int) -> float:
        return self.biomarker_values[biomarker][stage]

    def plot_disease_progression(self):
        plt.figure()
        for marker in self.model_values:
            plt.plot(marker)
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.xticks(ticks=np.linspace(0, len(self.model_values[0])-1, self.n_biomarker_stages),
                   labels=[f'{i+1}' for i in range(self.n_biomarker_stages)])
        plt.legend([f'Biomarker {i+1}' for i in range(self.n_biomarker_stages)])
        plt.show()