import numpy as np
from scipy.linalg import fractional_matrix_power

from model_generator.base_disease_model import BaseDiseaseModel
#from model_generator.biomarker_utils import generate_transition_matrix, initialize_biomarkers, simulate_progression_over_stages, compute_log_prior_from_adjacency

class TransitionMatrixModel(BaseDiseaseModel):
    def __init__(self, matrix_type='default', coeff=0.5, _lambda = 1.0, **params):
        super().__init__(**params)
        self.matrix_type = matrix_type
        self.coeff = coeff

    def fit(self, X=None, y=None):
        """Generate Transition Matrix model values."""
        n_stages = self.params.get('n_stages', 10)
        timespan = np.arange(self.steps)
        n_biomarkers = n_stages + 1  # generate extra to remove later
        transition_matrix = generate_transition_matrix(n_biomarkers, self.coeff)
        self.connectivity_matrix = compute_log_prior_from_adjacency(transition_matrix)
        y_init = initialize_biomarkers(n_biomarkers)

        self.model_values = simulate_progression_over_stages(transition_matrix, timespan, y_init).T[1:]

        return self
    
    def get_connectivity_matrix(self):
        return self.connectivity_matrix

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



import sys
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])
 
    # A utility function to find the vertex with minimum distance value, from the set of vertices
    def minDistance(self, dist, sptSet):
        min = sys.maxsize
        for u in range(self.V):
            if dist[u] < min and sptSet[u] == False:
                min = dist[u]
                min_index = u
        return min_index
 
    # Function that implements Dijkstra's single source shortest path algorithm for a graph represented
    def dijkstra(self, src):
 
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
            x = self.minDistance(dist, sptSet)
            sptSet[x] = True
            for y in range(self.V):
                if self.graph[x][y] > 0 and sptSet[y] == False and \
                        dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]
 
        return dist

# This code is contributed by Divyanshu Mehta and Updated by Pranav Singh Sambyal

def compute_log_prior_from_adjacency(A, _lambda = 1.0): # lambda is a reserved token!
    g = Graph(np.shape(A)[0])

    L = A.copy()
    L[L==0.0] = -1.0
    L = 1/L
    L[L<0.0] = 0.0

    g.graph = L
    log_prior = np.full(np.shape(A),0.0) #Only for tri-diagnoal A with constant off-diagnoal a

    for i in range(np.shape(A)[0]):
        dist = np.asarray(g.dijkstra(i))
    #     dist[dist == 0] = -1.0
        log_prior[:,i] = -1.0*dist

    # log_prior[log_prior > 0.0] = 0.0
    return log_prior * _lambda

