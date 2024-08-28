import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple 
from .biomarker_utils import generate_transition_matrix, initialize_biomarkers, simulate_progression_over_stages, Graph, compute_log_prior_from_adjacency
from .ode_generator import ODEGenerator

class CanonicalGenerator:
    def __init__(self, n_biomarker_stages: int, model_type: str = 'sigmoid', biomarkers_params: Dict = None, random_state: int = 10):
        """
        Initializes the CanonicalGenerator with the specified parameters. Parameters will create the biomarker progressions
        and then store them in model_values matrix and biomarker_values. model_values is the higher resolution representation
        of the biomarkers. `biomarker_values` are strictly the value of the biomarkers only at discrete stages, this is used as a
        lookup matrix for patient generation.
        
        Parameters:
            n_biomarker_stages (int): The number of stages/biomarkers to simulate.
            model_type (str): The model type used for generating biomarkers progressions. Ex. "sigmoid", "transition_matrix", "ode".
            biomarker_params (Dict):
        
        Attributes:
            model_values (np.ndarray): High-resolution representation of biomarker progressions.
            biomarker_values (np.ndarray): Discrete stage values for biomarker values used in patient generation.
        """
        self.n_biomarker_stages = n_biomarker_stages
        self.model_type = model_type
        self.biomarkers_params = biomarkers_params
        self.random_state = random_state
        
        # Attributes
        self.prior = None
        self.time_points = None
        self.model_values = self._generate_model()
        self.stage_values = self._get_discrete_stage_values() # values at each discrete stage

    def _generate_model(self) -> np.ndarray:
        """
        Calls the appropriate model generation function depending on `model_type` selected.
        
        Returns:
            np.ndarray: M-by-N matrix where each row is a different biomarker, and column is the value of a biomarker at an arbitrary time point
        """
        if self.model_type == 'sigmoid':
            return self._generate_sigmoid_model()
        elif self.model_type == 'transition_matrix':
            return self._generate_transition_matrix_model()
        elif self.model_type == 'logistic':
            return self._generate_ode_model()
        elif self.model_type == 'acp':
            return self._generate_ode_model()
        elif self.model_type == 'diffusion':
            return self._generate_ode_model()
        elif self.model_type == 'reaction_diffusion':
            return self._generate_ode_model()
        # elif self.model_type == 'example':
        #     return self._generate_example_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def _biomarker_sort(self, model_values: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        markers_sorted = 0
        column_idx = 0
        while markers_sorted < model_values.shape[0]:
            values = model_values[markers_sorted:, column_idx]
            if column_idx == model_values.shape[1]-1:
                return model_values
            if np.max(values) > threshold:
                max_index = np.where(model_values[:,column_idx] == np.max(values))
                model_values[max_index, :], model_values[markers_sorted, :] = model_values[markers_sorted, :]
                markers_sorted += 1
            column_idx += 1
        return model_values

    def _generate_sigmoid_model(self) -> np.ndarray:
        def sigmoid(x, s, c):
            return 1 - 1 / (1 + np.exp(-s * (x - c)))

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
            The generated transition matrix model as an np.ndarray
        """
        coeff = self.biomarkers_params.get('coeff')
        timespan = np.arange(40)
        n_biomarkers = self.n_biomarker_stages + 1 # generate extra to remove later
        
        transition_matrix = generate_transition_matrix(n_biomarkers, coeff)
        self.prior = compute_log_prior_from_adjacency(transition_matrix)
        y_init = initialize_biomarkers(n_biomarkers)
        model_values = simulate_progression_over_stages(transition_matrix, timespan, y_init)
        return model_values.T[1:]
    

    
    # TODO: make mult_logistic_sym a parameter
    # TODO: add param grid for initial conditions
    def _generate_ode_model(self) -> np.ndarray:
        def _biomarker_sort(model_values: np.ndarray, threshold: float = 0.50) -> np.ndarray:
            """
            WARNING! this is a temporary solution for the out of order biomarker generation for the ODE model.
            Reorders biomarkers based on which marker reaches top values first.
            """
            markers_sorted = 0
            column_idx = 0
            while markers_sorted < model_values.shape[0]:
                values = model_values[markers_sorted:, column_idx]
                if column_idx == model_values.shape[1]-1:
                    return model_values
                if np.max(values) > threshold:
                    max_index = np.where(model_values[:,column_idx] == np.max(values))
                    model_values[max_index, :], model_values[markers_sorted, :] = model_values[markers_sorted, :], model_values[max_index, :]
                    markers_sorted += 1
                column_idx += 1
            return model_values
        
        def _normalize_model_values(model_values: np.ndarray) -> np.ndarray:
            """
            Normalize all biomarker values so that the maximum value across all biomarkers is 1.
            """
            max_value = np.max(model_values)  # Find the global maximum across all biomarkers
            return model_values / max_value  # Normalize all values by the global maximum
        
        ode_generator = ODEGenerator(self.n_biomarker_stages, model_type=self.model_type, random_state = self.random_state)
        t, x = ode_generator.generate_model()
        self.time_points = t
        self.prior = ode_generator.get_connectivity_matrix()
        x = _biomarker_sort(x)
        x = _normalize_model_values(x)
        return x # x is just the model_value at each timepoint t
    
    def _normalize_model_values(model_values: np.ndarray) -> np.ndarray:
        """
        Normalize each biomarker's values so that the maximum value is 1.
        """
        max_values = np.max(model_values, axis=1, keepdims=True)
        return model_values / max_values
    
    def _get_discrete_stage_values(self) -> None: # this could be better written
        """
        Converts `self.model_values` -> `self.stage_values` (nd.array) 
        by taking the value of each biomarker at each stage.
        """
        if self.model_type in ['logistic']:
            for i in range(len(self.model_values)):
                print(f"self.time_points length: {len(self.time_points)}, self.model_values[{i}] length: {len(self.model_values[i])}")
                if len(self.time_points) != len(self.model_values[i]):
                    raise ValueError(f"Length mismatch: time_points={len(self.time_points)}, model_values[{i}]={len(self.model_values[i])}")
            return np.array([np.interp(np.arange(self.n_biomarker_stages), self.time_points, self.model_values[i]) for i in range(len(self.model_values))])
        else:
            return self.model_values[:, np.linspace(0, self.model_values.shape[1] - 1, self.n_biomarker_stages, dtype=int)]
    
    ## Getters
    
    def get_stage_values(self) -> np.ndarray: # TODO: discuss renaming to "generate_model" or "generate"
        """Gets values at each discrete stage"""
        return self.stage_values
    
    def get_model_values(self) -> np.ndarray:
        """Gets all values from models over model domain"""
        return self.model_values
        
    def get_prior(self) -> np.ndarray:
        """Gets prior."""
        return self.prior
    
    
    ## Plotting

    def plot_disease_progression(self):
        """
        Plots all biomarker progressions with stages visible.
        """
        plt.figure()
        for marker in self.model_values:
            plt.plot(marker)
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.xticks(ticks=np.linspace(0, len(self.model_values[0])-1, self.n_biomarker_stages),
                   labels=[f'{i+1}' for i in range(self.n_biomarker_stages)])
        plt.legend([f'Biomarker {i+1}' for i in range(self.n_biomarker_stages)])
        plt.show()