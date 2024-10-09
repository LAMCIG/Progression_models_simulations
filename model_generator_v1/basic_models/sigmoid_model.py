import numpy as np
from typing import Dict, Any
from model_generator_v1.base_model import BaseModel  # <-- This is the missing import

class SigmoidModel(BaseModel):
    def __init__(self, n_stages: int, biomarker_params: Dict[int, Dict[str, float]], **kwargs):
        """
        Initializes the SigmoidModel.

        Parameters:
        - n_stages (int): Number of biomarkers.
        - biomarker_params (Dict): Dictionary containing parameters for each biomarker.
        """
        super().__init__(n_stages)
        self.biomarker_params = biomarker_params
        self.model_values = self.generate_model()

    def sigmoid(self, x, s, c):
        """sigmoid function."""
        return 1 / (1 + np.exp(-s * (x - c)))

    def generate_model(self) -> np.ndarray:
        """
        Generates the model values for each biomarker based on sigmoid functions.
        """
        x = np.linspace(-50, 120, 1000)
        model_values = np.zeros((self.n_stages, len(x)))

        for biomarker, params in self.biomarker_params.items():
            s = params['s']
            c = params['c']
            model_values[biomarker] = self.sigmoid(x, s, c)

        return model_values

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters for the model."""
        return self.biomarker_params

    def set_parameters(self, biomarker_params: Dict[int, Dict[str, float]]):
        """Updates the parameters and regenerates the model."""
        self.biomarker_params = biomarker_params
        self.model_values = self.generate_model()
