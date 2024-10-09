from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BaseDiseaseModel:
    def __init__(self, normalize=True, flip_v=False, flip_h=False, **params):
        self.normalize = normalize
        self.flip_v = flip_v
        self.flip_h = flip_h
        self.params = params
        self.model_values = None
    
    def fit(self, X=None, y=None):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def transform(self, X=None):
        """Apply the transformation (return model values and apply utilities)."""
        if self.model_values is None:
            raise ValueError("Model values need to be generated. Call fit() first.")
        
        output = self.model_values
        if self.normalize:
            output = self._normalize(output)
        if self.flip_v:
            output = np.flipud(output)
        if self.flip_h:
            output = np.fliplr(output)
        
        return output

    def _normalize(self, values):
        """Utility to normalize the model values between 0 and 1."""
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
