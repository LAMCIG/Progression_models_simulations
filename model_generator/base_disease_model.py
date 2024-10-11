from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt

class BaseDiseaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, start_time=0, end_time=100, steps=100, staging='inset',
                 normalize=True, flip_v=False, flip_h=False, convergence_threshold=1e-2, **params):
        """
        Base class for disease models with common utilities.

        Parameters:
        - start_time (float): The starting point of the time domain.
        - end_time (float): The end point of the time domain.
        - steps (int): Number of steps (time points) between start_time and end_time.
        - discrete (bool): Whether the model should return discrete stages. Default is True.
        - normalize (bool): Whether to normalize the output between 0 and 1.
        - flip_v (bool): Whether to flip the model output vertically.
        - flip_h (bool): Whether to flip the model output horizontally.
        - convergence_threshold (float): Threshold to determine when biomarkers are stable.
        - params (dict): Additional parameters for the specific disease model.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.steps = steps
        self.staging = staging
        self.normalize = normalize
        self.flip_v = flip_v
        self.flip_h = flip_h
        self.convergence_threshold = convergence_threshold
        self.params = params
        self.model_values = None
        self.original_model_values = None  # store continuous model values
    
    def fit(self, X=None, y=None):
        """Creates raw model values before transform"""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def transform(self, X=None):
        """Applies optional utilities to modify and preprocess data"""
        if self.model_values is None:
            raise ValueError("Model values need to be generated. Call fit() first.")
        
        self.original_model_values = self.model_values.copy()
        output = self.model_values
        
        # determine stage definition: whole, inset, or dynamic
        if self.staging == 'inset':
            output = self._get_static_stage_values(output)
        elif self.staging == 'dynamic':
            output = self._get_dynamic_stage_values(output)
        else:  # default to whole
            output = self._get_discrete_stage_values(output)
        
        # apply rest
        if self.normalize:
            output = self._normalize(output)
        if self.flip_v:
            output = 1 - output
        if self.flip_h:
            output = np.fliplr(output)
        
        return output

    def _normalize(self, values):
        """Utility to normalize the model values between 0 and 1."""
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
    
    def _get_discrete_stage_values(self, values):
        """Return the biomarker values at discrete stages based on the number of biomarkers (n_stages)."""
        n_stages = self.params.get('n_stages', 10)
        discrete_indices = np.linspace(0, values.shape[1] - 1, n_stages, dtype=int)  # evenly spaced indices
        return values[:, discrete_indices]
    
    def _get_static_stage_values(self, values, inset_ratio=0.1):
        """Returns biomarker values with evenly spaced stages between the start and end, excluding static points."""
        n_stages = self.params.get('n_stages', 10)
        num_time_points = values.shape[1]
    
        inset_start = int(inset_ratio * num_time_points)
        inset_end = int((1 - inset_ratio) * num_time_points)
        stage_indices = np.linspace(inset_start, inset_end, n_stages, dtype=int)
        
        return values[:, stage_indices]

    def _get_dynamic_stage_values(self, values, threshold=0.5):
        # TODO: THIS IS BROKEN
        """Dynamically assigns stages based on biomarker crossing a threshold."""
        n_stages = self.params.get('n_stages', 10)
        dynamic_stages = []

        for biomarker in values:
            stage_indices = np.where(np.abs(biomarker - threshold) < 0.01)[0] # first one only
            dynamic_stages.append(stage_indices[:n_stages])
        
        return np.array(dynamic_stages)

    def _dynamic_time_adjustment(self, values):
        
        changes = np.abs(np.diff(values, axis=1))  # like MATLAB diff, maek positive
        max_change = np.max(changes, axis=1)
        if np.all(max_change < self.convergence_threshold):
            return np.linspace(self.start_time, self.end_time, self.steps)
        else:
            self.end_time *= 1.5  # naively just increase end_time
            self.steps = int(self.steps * 1.5)
            return np.linspace(self.start_time, self.end_time, self.steps)

    def plot(self, expand_ratio=0.):
        
        if self.model_values is None:
            raise ValueError("Model values need to be generated. Call fit() first.")
        
        self.discrete = False  # temporarily disable discretization for smooth plotting
        output = self.transform(X = None)  # apply transformations (normalize, flip, etc.) without discretizing
        self.discrete = True  # restore discretization for other uses if necessary
    
        for i, marker in enumerate(output):
            plt.plot(marker, label=f'Biomarker {i+1}')
        
        # set x-ticks to reflect the discrete stages
        n_stages = self.params.get('n_stages', 10)
        stage_ticks = np.linspace(0, len(output[0]) - 1, n_stages)
        plt.xticks(ticks=stage_ticks, labels=[f'{i+1}' for i in range(n_stages)])
        
        # expand plot edges while keeping ticks fixed
        xmin, xmax = 0, len(output[0]) - 1
        x_range = xmax - xmin
        plt.xlim(xmin - expand_ratio * x_range, xmax + expand_ratio * x_range)
        
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.show()
