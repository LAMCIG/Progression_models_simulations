import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.integrate import solve_ivp
import random
import sys

#%% SIGMOID UTILS

def sigmoid_inv(x: float, s: float = 0, c: float = 1) -> float:
    """
    Calculates the inverse of a sigmoid function to model biomarker progression.
    
    Parameters:
    x (float): The input value to the sigmoid function.
    s (float): The shift parameter, adjusting the function along the x-axis.
    c (float): The scale parameter, adjusting the steepness of the sigmoid curve.
    
    Returns:
    float: The calculated inverse sigmoid value.
    """
    return 1 - 1 / (1 + np.exp(-(x-s)/c))


#%% TRANSITION MATRIX UTILS

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

#%% ODE UTILS

def get_adjacency_matrix(matrix_type, size):
    if matrix_type == "Random Zero Diagonal":
        return random_zero_diagonal_matrix(size)
    elif matrix_type == "Binary Zero Diagonal":
        return binary_zero_diagonal_matrix(size)
    elif matrix_type == "Tridiagonal":
        return tridiagonal_matrix(size)
    else:
        return tridiagonal_matrix(size)  # default to tridiagonal


def random_connectivity_matrix(n, med_frac, source_rate, all_source_connections):
    random.seed(10)
    A = np.random.rand(n, n)
    A = np.dot(A.T, A)
    np.fill_diagonal(A, 0)
    K = np.zeros((n, n))
    K = A.copy()
    
    for i in range(n):
        loc_thresh = min(med_frac * np.median(A[i, 1:]), max(A[i, 1:] * 0.99))
        ind = np.where(A[i, 1:] < loc_thresh)[0] + 1
        K[i, ind] = 0
        K[ind, i] = 0

    S = np.sum(K > 0, axis=0)
    for i in range(n):
        if S[i] == 0 or (i == 1 and S[i] < 2):
            if i != 1:
                ind = np.argmax(A[i, 1:]) + 1
                K[i, ind] = A[i, ind]
                K[ind, i] = A[ind, i]
            else:
                ind = np.argmax(A[i, 2:]) + 2
                K[i, ind] = A[i, ind]
                K[ind, i] = A[ind, i]

    K = K / np.max(K)
    
    if all_source_connections:
        K[0, :] = source_rate
        K[:, 0] = source_rate
    else:
        K[0, :] = 0
        K[:, 0] = 0

    K[0, 1] = source_rate
    K[1, 0] = source_rate
    
    return K

def multi_logistic_deriv(t, x, K):
    return (np.eye(len(K)) - np.diag(x)) @ K @ x

def solve_ode_system(K, x0, t_span, n_steps):
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(lambda t, x: multi_logistic_deriv(t, x, K), t_span, x0, t_eval=t_eval)
    return sol.t, sol.y.T

def random_zero_diagonal_matrix(size, random_state = 10):
    """Generates an adjacency matrix with zero diagonal."""
    rng = np.random.RandomState(random_state) # fixed random state
    A = rng.rand(size, size)
    np.fill_diagonal(A, 0)
    return A

def binary_zero_diagonal_matrix(size):
    """Generates an adjacency matrix with zero diagonal."""
    A = np.ones([size, size])
    np.fill_diagonal(A, 0)
    return A

def tridiagonal_matrix(size):
    """Generates a binary tridiagonal adjacency matrix with zero diagonal."""
    A = np.zeros((size, size))
    np.fill_diagonal(A[:-1, 1:], 1)
    np.fill_diagonal(A[1:, :-1], 1)
    return A

def compute_laplacian_matrix(A):
    """Generates the Laplacian matrix from the adjacency matrix as described by Garbarino."""
    degree_matrix = np.diag(np.sum(A, axis=1))
    laplacian_matrix = degree_matrix - A
    return laplacian_matrix



#%%
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