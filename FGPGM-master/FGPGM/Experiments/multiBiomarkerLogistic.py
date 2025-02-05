# FGPGM/Experiments/multiBiomarkerLogistic.py

from ..Experiment import Experiment
import numpy as np

def get_adjacency_matrix(connectivity_matrix_type, n_biomarkers):
    if connectivity_matrix_type == 'offdiag':
        matrix = np.zeros((n_biomarkers, n_biomarkers))
        for i in range(n_biomarkers - 1):
            matrix[i, i + 1] = 1
            matrix[i + 1, i] = 1
    else:
        raise ValueError("Unknown connectivity matrix type")
    return matrix

class MultiBiomarkerLogistic(Experiment):
    def __init__(self, n_biomarkers=5, connectivity_matrix_type='offdiag'):
        super(MultiBiomarkerLogistic, self).__init__()
        self.n_biomarkers = n_biomarkers
        self.connectivity_matrix_type = connectivity_matrix_type
        self.K = get_adjacency_matrix(connectivity_matrix_type, n_biomarkers)
    
    def f(self, x, theta):
        # dx/dt = (I - diag(x)) @ (Kx + f)
        n = self.n_biomarkers
        f = np.zeros(self.n_biomarkers) # theta[:n]
        f[0] = 0.01 
        Kx = self.K.dot(x)
        Id_minus_diag = np.eye(n) - np.diag(x)
        dx_dt = Id_minus_diag.dot(Kx + f)
        return dx_dt

    def getBounds(self, nStates, nParams, x0=None):
        xmin = []
        xmax = []
        for i in range(nStates):
            xmin.append(0)
            xmax.append(2)
        for i in range(nParams):
            xmin.append(-1)
            xmax.append(2)
        return xmin, xmax

def generate_logistic_model(n_biomarkers=5, step=0.1, n_steps=100, neg_frac=0.01, connectivity_matrix_type='offdiag'):
    t0 = -neg_frac * n_steps * step
    t = np.arange(t0, (n_steps + 1) * step, step)
    
    x0 = np.zeros(n_biomarkers)
    f = np.zeros(n_biomarkers)
    f[0] = 0.01
    
    x = np.zeros((n_biomarkers, len(t)))
    zero_ind = np.where(t == 0)[0][0]
    x[:, zero_ind] = x0

    K = get_adjacency_matrix(connectivity_matrix_type, n_biomarkers)
    
    for i in range(zero_ind - 1, -1, -1):
        force = np.zeros_like(f)
        dx_dt = (np.eye(n_biomarkers) - np.diag(x[:, i + 1])).dot(K.dot(x[:, i + 1]) + force)
        x[:, i] = np.maximum(0, x[:, i + 1] - dx_dt * step)

    for i in range(zero_ind, n_steps + zero_ind):
        force = np.exp(t[i] * f) - 1
        dx_dt = (np.eye(n_biomarkers) - np.diag(x[:, i])).dot(K.dot(x[:, i]) + force)
        x[:, i + 1] = x[:, i] + dx_dt * step

    return t, x, K
