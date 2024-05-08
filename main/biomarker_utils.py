import numpy as np
from scipy.linalg import fractional_matrix_power

def sigmoid_inv(x, s=0, c=1):
    return 1 - 1 / (1 + np.exp(-(x-s)/c))

def generate_transition_matrix(size, coeff):
    """Generates a symmetric transition matrix with coefficient decay off-diagonal."""
    A = np.eye(size)
    np.fill_diagonal(A[:-1, 1:], coeff)
    np.fill_diagonal(A[1:, :-1], coeff)
    return A.astype(np.float32)

def initialize_biomarkers(num_biomarkers, init_value=0.9):
    """Initializes biomarkers with a fixed initial value for the first biomarker."""
    y = np.full(num_biomarkers, 1, dtype=np.float32)
    y[0] = init_value
    return y

def apply_transition_matrix(A, n, y_init):
    """Applies the transition matrix to simulate biomarker progression for n stages."""
    y = np.clip(A @ (1 - y_init), 0, 1)
    for _ in range(1, n):
        y = np.clip(A @ y, 0, 1)
    return 1 - y

def softplus(x): # added in to smooth the biomarker curves
    return np.log(1.0+np.exp(x))

def apply_transition_matrix2(A, n, y_init, alpha = 10, delta = 0.5):

    y = 1 - y_init 
    if(len(np.shape(n)) == 0):
        y_out = np.clip(fractional_matrix_power(A,n)@y,0,1)
      
    else:
        y_out = np.full([np.shape(y)[0],np.shape(n)[0]],1.0)
        for s,j in zip(n,range(len(n))):
            y_out[:,j] = np.clip(fractional_matrix_power(A,s)@y,0,1)
    y = 1 - y_out   
    
    y = softplus(alpha*(y-delta)) / softplus(alpha*(1.0-delta))
              
    return y

def simulate_progression_over_stages(transition_matrix, stages, y_init):
    """Simulates the progression of biomarker values over multiple stages."""
    return np.array([apply_transition_matrix(transition_matrix, stage, y_init) for stage in stages])

def simulate_progression_over_stages2(transition_matrix, stages, y_init):
    """Simulates the progression of biomarker values over multiple stages."""
    return np.array([apply_transition_matrix2(transition_matrix, stage, y_init) for stage in stages])