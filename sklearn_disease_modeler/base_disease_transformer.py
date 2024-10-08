from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BaseDiseaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, connectivity_matrix=None, normalize=True, **params):
        self.connectivity_matrix = connectivity_matrix
        self.params = params
        self.normalize = normalize
        self.model_values = None

    def fit(self, X=None, y=None):
        # Subclasses will implement their own fitting (model generation)
        raise NotImplementedError("Subclasses must override fit to generate model values.")

    def transform(self, X=None):
        if self.model_values is None:
            raise ValueError("The model needs to be fit first.")
        if self.normalize:
            return self.normalize_values(self.model_values)
        return self.model_values

    def get_parameters(self):
        return self.params

    def set_parameters(self, params):
        self.params = params

    def normalize_values(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
