import numpy as np
from scipy.linalg import fractional_matrix_power

from model_generator.base_disease_model import BaseDiseaseModel
from model_generator.biomarker_utils import generate_transition_matrix, initialize_biomarkers, simulate_progression_over_stages, compute_log_prior_from_adjacency

class TransitionMatrixModel(BaseDiseaseModel):
    def __init__(self, matrix_type='default', coeff=0.5, **params):
        super().__init__(**params)
        self.matrix_type = matrix_type
        self.coeff = coeff

    def fit(self, X=None, y=None):
        """Generate Transition Matrix model values."""
        n_stages = self.params.get('n_stages', 10)
        timespan = np.arange(self.steps)
        n_biomarkers = n_stages + 1  # generate extra to remove later
        transition_matrix = generate_transition_matrix(n_biomarkers, self.coeff)
        self.prior = compute_log_prior_from_adjacency(transition_matrix)
        y_init = initialize_biomarkers(n_biomarkers)

        self.model_values = simulate_progression_over_stages(transition_matrix, timespan, y_init).T[1:]

        return self

def generate_transition_matrix(size: int, coeff: float) -> np.ndarray:
    """
    Generates a symmetric transition matrix with a specified coefficient decay off-diagonal.
    
    Parameters:
    size (int): The size of the square matrix to generate.
    coeff (float): The coefficient value to fill off-diagonal adjacency positions.
    
    Returns:
    np.ndarray: A symmetric N by N transition matrix of specified size and coefficients.
    """
    A = np.eye(size)
    np.fill_diagonal(A[:-1, 1:], coeff)
    np.fill_diagonal(A[1:, :-1], coeff)
    return A.astype(np.float32)

def initialize_biomarkers(num_biomarkers, init_value=0.9):
    """Initializes biomarkers with a fixed initial value for the first biomarker."""
    y = np.full(num_biomarkers, 1, dtype=np.float32)
    y[0] = init_value
    return y

def softplus(x):
    """Softplus activation function. For smoother curves."""
    return np.log(1.0 + np.exp(x))

def apply_transition_matrix(A, stages, y_init, alpha=10, delta=0.5):
    """Applies the transition matrix to simulate biomarker progression for n stages."""
    y = 1 - y_init
    if len(np.shape(stages)) == 0:
        y_out = np.clip(fractional_matrix_power(A, stages) @ y, 0, 1)
    
    else:
        y_out = np.full([np.shape(y)[0], np.shape(stages)[0]], 1.0)
        for s, j in zip(stages, range(len(stages))):
            y_out[:, j] = np.clip(fractional_matrix_power(A, s) @ y, 0, 1)
    y = 1 - y_out
    
    y = softplus(alpha * (y - delta)) / softplus(alpha * (1.0 - delta))
    
    return y

def simulate_progression_over_stages(transition_matrix, stages, y_init):
    """Simulates the progression of biomarker values over multiple stages."""
    return np.array([apply_transition_matrix(transition_matrix, stage, y_init) for stage in stages])