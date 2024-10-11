import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.integrate import solve_ivp
import random
import sys

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