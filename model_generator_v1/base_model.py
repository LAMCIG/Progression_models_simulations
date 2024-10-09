from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseModel(ABC):
    def __init__(self, n_stages: int, **params):
        """
        Base class for all models.

        Parameters:
        - n_stages (int): Number of stages (biomarkers).
        - params (Dict): Additional parameters for the model.
        """
        self.n_stages = n_stages
        self.model_values = None

    @abstractmethod
    def generate_model(self) -> np.ndarray:
        """Generates the model values as a numpy array."""
        pass

    def get_stage_values(self) -> np.ndarray:
        """Get discrete stage values."""
        if self.model_values is not None:
            return self.model_values[:, np.linspace(0, self.model_values.shape[1] - 1, self.n_stages, dtype=int)]
        else:
            raise NotImplementedError("Model values must be generated first.")
        
    def get_model_values(self) -> np.ndarray:
        """Get model values."""
        return self.model_values

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters used by the model."""
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]):
        """Updates model parameters and regenerates the model."""
        pass
