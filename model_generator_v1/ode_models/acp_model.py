import numpy as np
from typing import Dict, Any
from model_generator_v1.base_model import BaseModel
from model_generator_v1.biomarker_utils import get_adjacency_matrix, compute_laplacian_matrix


class ACPModel(BaseModel):
    def __init__(self, n_stages: int, adjacency_matrix_type: str, l1_mean: float, l2_mean: float, gamma_mean: float, eta_mean: float, k_ij_value: float, random_state: int = 10, **kwargs):
        super().__init__(n_stages)
        self.adjacency_matrix_type = adjacency_matrix_type
        self.l1_mean = l1_mean
        self.l2_mean = l2_mean
        self.gamma_mean = gamma_mean
        self.eta_mean = eta_mean
        self.k_ij_value = k_ij_value
        self.random_state = random_state

        self.model_values = self.generate_model()

    def acp_equations(self, f, tau, A, H, k_ij, gamma, eta, l1, l2):
        exp_1 = -l1 * (f - gamma)
        exp_2 = l2 * (f - eta)

        K_ACP = k_ij / ((1 + np.exp(exp_1)) * (1 + np.exp(exp_2)))
        R_ACP = k_ij / (1 + np.exp(exp_2))
        dfdtau = np.dot((A * K_ACP), np.dot(H, f)) + np.multiply(R_ACP, f)
        return dfdtau

    def generate_model(self) -> np.ndarray:
        A = get_adjacency_matrix(self.adjacency_matrix_type, self.n_stages)
        H = compute_laplacian_matrix(A)
        
        np.random.seed(self.random_state)
        n = A.shape[0]
        t = np.linspace(0, 20, 1000)
        dt = t[1] - t[0]  # time step
        x0 = np.zeros(n)
        x0[0] = 0.05

        # parameters
        l1 = np.random.normal(loc=self.l1_mean, scale=1, size=n)
        l2 = np.random.normal(loc=self.l2_mean, scale=1, size=n)
        gamma = np.random.normal(loc=self.gamma_mean, scale=0.1, size=n)
        eta = np.random.normal(loc=self.eta_mean, scale=0.1, size=n)
        k_ij = np.random.normal(loc=self.k_ij_value, scale=0.1, size=n)

        # solution array
        x = np.zeros((n, len(t)))
        x[:, 0] = x0

        # forward euler method
        for i in range(1, len(t)):
            dx_dt = self.acp_equations(x[:, i-1], t[i-1], A, H, k_ij, gamma, eta, l1, l2)
            x[:, i] = x[:, i-1] + dx_dt * dt
            x[:, i] = np.maximum(x[:, i], 0)  # enforce non-negativity

        return x

    def get_parameters(self) -> dict:
        return {
            'adjacency_matrix_type': self.adjacency_matrix_type,
            'l1_mean': self.l1_mean,
            'l2_mean': self.l2_mean,
            'gamma_mean': self.gamma_mean,
            'eta_mean': self.eta_mean,
            'k_ij_value': self.k_ij_value,
            'random_state': self.random_state
        }

    def set_parameters(self, params: dict):
        self.adjacency_matrix_type = params.get('adjacency_matrix_type', self.adjacency_matrix_type)
        self.l1_mean = params.get('l1_mean', self.l1_mean)
        self.l2_mean = params.get('l2_mean', self.l2_mean)
        self.gamma_mean = params.get('gamma_mean', self.gamma_mean)
        self.eta_mean = params.get('eta_mean', self.eta_mean)
        self.k_ij_value = params.get('k_ij_value', self.k_ij_value)
        self.random_state = params.get('random_state', self.random_state)
        self.model_values = self.generate_model()
