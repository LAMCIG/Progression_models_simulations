import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import fractional_matrix_power


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

def generate_patient(stage, transition_matrix, y_init, add_noise=False, noise_std=0.1, random_state=None):
    """Generates vector of patient biomarkers given patient stage. Optionally adds Gaussian noise to biomarkers.
    
    Parameters:
    - stage: The stage of AD for the patient.
    - transition_matrix: The transition matrix used for simulating progression.
    - y_init: Initial biomarker values.
    - add_noise: If True, adds Gaussian noise to the biomarker values.
    - noise_std: Standard deviation of the Gaussian noise.
    - random_state: Seed for the random number generator for reproducibility.
    """
    y_stage = apply_transition_matrix2(transition_matrix, stage, y_init)  # Simulate progression to given stage
    if add_noise:
        random = np.random.RandomState(random_state)
        noise = random.normal(0, noise_std, y_stage.size)
        y_stage += noise
    y_stage = np.clip(y_stage, 0, 1)  # Ensure biomarker values are within [0, 1]
    return y_stage

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


#%%

num_biomarkers = 11
A = generate_transition_matrix(size=num_biomarkers, coeff=1e-1)
A = A[1:, 1:]  # remove the first row and first column

prior = compute_log_prior_from_adjacency(A)
#prior = prior[1:, 1:]  # adjusting the prior matrix similarly
np.shape(prior)

y_init = initialize_biomarkers(num_biomarkers, init_value=0.9)
y_init = y_init[1:]  # remove the first element
y_init[0] = 0.9  # remove the first element

biomarkers_params = {
    'transition_matrix': A,
    'y_init': y_init
}

biomarkers_params